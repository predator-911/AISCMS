import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from io import StringIO
import pandas as pd
import numpy as np
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
import certifi
from enum import Enum
from heapq import heappush, heappop
import uvicorn

# ---------------------------- Configuration ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("space-cargo")

# MongoDB Configuration
MONGODB_USERNAME = os.getenv("MONGODB_USERNAME")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
MONGO_DB = os.getenv("MONGO_DB")

# ✅ Connect to MongoDB Atlas
uri = f"mongodb+srv://{MONGODB_USERNAME}:{MONGO_PASSWORD}@cluster0.38cb2.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, tlsCAFile=certifi.where())
db = client[MONGO_DB]
cargo_collection = db["cargo"]
log_collection = db["logs"]
storage_containers = db["storage_containers"]
items_col = db["items"]  # Added missing collection

# Time management
CURRENT_TIME = datetime.now().isoformat()

def get_current_time() -> str:
    global CURRENT_TIME
    return CURRENT_TIME

def set_current_time(new_time: str) -> None:
    global CURRENT_TIME
    CURRENT_TIME = new_time

# ---------------------------- Data Models ----------------------------
class ItemRotation(str, Enum):
    WDH = "wdh"
    WHD = "whd"
    DWH = "dwh"
    DHW = "dhw"
    HWD = "hwd"
    HDW = "hdw"

class Coordinates(BaseModel):
    width: float = Field(..., ge=0)
    depth: float = Field(..., ge=0)
    height: float = Field(..., ge=0)

class Item(BaseModel):
    itemId: str
    name: str
    width: float
    depth: float
    height: float
    mass: float
    priority: int = Field(..., ge=1, le=100)
    expiryDate: Optional[str] = None  # ISO format
    usageLimit: Optional[int] = None
    preferredZone: str

class Container(BaseModel):
    containerId: str
    zone: str
    width: float
    depth: float
    height: float

class PlacementRequest(BaseModel):
    items: List[Item]
    containers: List[Container]

class PlacementResponse(BaseModel):
    success: bool
    placements: List[Dict]
    rearrangements: List[Dict]

class RetrievalStep(BaseModel):
    step: int
    action: str  # "remove", "retrieve", "placeBack"
    itemId: str
    itemName: str

class WasteItem(BaseModel):
    itemId: str
    name: str
    reason: str  # "expired" or "depleted"
    containerId: str
    position: Dict

class ReturnPlanResponse(BaseModel):
    success: bool
    totalWeight: float
    totalVolume: float
    steps: List[Dict]

class RetrieveRequest(BaseModel):
    itemId: str
    userId: str
    timestamp: str

class PlaceRequest(BaseModel):
    itemId: str
    userId: str
    timestamp: str
    containerId: str
    position: Dict

class SimulationRequest(BaseModel):
    numOfDays: Optional[int] = None
    toTimestamp: Optional[str] = None
    itemsToBeUsedPerDay: List[Dict[str, str]]

class LogQuery(BaseModel):
    startDate: str
    endDate: str
    itemId: Optional[str] = None
    userId: Optional[str] = None
    actionType: Optional[str] = None

# ---------------------------- 3D Bin-Packing Core ----------------------------
class GuillotineBin:
    def __init__(self, width: float, depth: float, height: float):
        self.width = width
        self.depth = depth
        self.height = height
        self.free_rects = [(0, 0, 0, width, depth, height)]
        self.placements = []

    @staticmethod
    def split_rect(rect, item_width, item_depth, item_height):
        """Split remaining space after placing an item"""
        x, y, z, w, d, h = rect
        remaining = []
        
        # Vertical split
        if w > item_width:
            remaining.append((x + item_width, y, z, w - item_width, d, h))
        
        # Horizontal split
        if d > item_depth:
            remaining.append((x, y + item_depth, z, w, d - item_depth, h))
        
        # Height split
        if h > item_height:
            remaining.append((x, y, z + item_height, w, d, h - item_height))
        
        return remaining

    def insert(self, item: Item, rotation: ItemRotation) -> Optional[Dict]:
        """Try to insert item with given rotation, return position if successful"""
        # Apply rotation
        if rotation == ItemRotation.WDH:
            iw, id_, ih = item.width, item.depth, item.height
        elif rotation == ItemRotation.WHD:
            iw, id_, ih = item.width, item.height, item.depth
        elif rotation == ItemRotation.DWH:
            iw, id_, ih = item.depth, item.width, item.height
        elif rotation == ItemRotation.DHW:
            iw, id_, ih = item.depth, item.height, item.width
        elif rotation == ItemRotation.HWD:
            iw, id_, ih = item.height, item.width, item.depth
        else:  
            iw, id_, ih = item.height, item.depth, item.width

        # Find best fit (minimize depth for accessibility)
        best_score = float('inf')
        best_rect = None
        best_idx = -1

        for idx, rect in enumerate(self.free_rects):
            x, y, z, w, d, h = rect
            if w >= iw and d >= id_ and h >= ih:
                # Score based on depth (lower y is better)
                score = y + id_  # Prefer items closer to front
                if score < best_score:
                    best_score = score
                    best_rect = rect
                    best_idx = idx

        if not best_rect:
            return None

        # Place the item
        x, y, z, w, d, h = best_rect
        placement = {
            "x": x,
            "y": y,
            "z": z,
            "width": iw,
            "depth": id_,
            "height": ih,
            "rotation": rotation
        }

        # Split remaining space
        del self.free_rects[best_idx]
        self.free_rects.extend(self.split_rect(best_rect, iw, id_, ih))
        
        self.placements.append(placement)
        return placement

# ---------------------------- Database Setup ----------------------------
def get_db():
    client = MongoClient(uri, tlsCAFile=certifi.where())
    try:
        db = client[MONGO_DB]
        return db
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")

def get_collection(name: str, db=None):
    if not db:
        db = get_db()
    return db[name]

# ---------------------------- FastAPI Setup ----------------------------
app = FastAPI(title="Space Cargo Management", version="1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database collections dependency
def get_items_collection():
    return get_collection("items")

def get_containers_collection():
    return get_collection("containers")

def get_logs_collection():
    return get_collection("logs")

# Initialize indexes on startup
@app.on_event("startup")
async def startup_db_client():
    try:
        db = get_db()
        items_col = db["items"]
        containers_col = db["containers"]
        logs_col = db["logs"]
        
        # Create indexes
        items_col.create_index([("priority", DESCENDING), ("expiryDate", ASCENDING)])
        containers_col.create_index([("zone", ASCENDING)])
        logs_col.create_index([("timestamp", DESCENDING)])
        
        logger.info("Database indexes created successfully")
    except Exception as e:
        logger.error(f"Failed to create database indexes: {e}")

# ---------------------------- API Endpoints ----------------------------

@app.post("/api/add_cargo")
async def add_cargo(
    item: Item,
    items_col: Collection = Depends(get_items_collection),
    logs_col: Collection = Depends(get_logs_collection)
):
    item_dict = item.dict()
    item_dict["expiryDate"] = (datetime.now() + timedelta(days=30)).isoformat()
    item_dict["usageCount"] = 0  # Initialize usage count
    item_dict["retrieval_count"] = 0
    item_dict["created_at"] = datetime.now().isoformat()
    item_dict["isWaste"] = False

    inserted_item = items_col.insert_one(item_dict)

    # Log the action
    logs_col.insert_one({
        "action": "add_cargo",
        "itemId": item.itemId,
        "timestamp": datetime.now().isoformat()
    })

    return {"item_id": str(inserted_item.inserted_id), "message": "Cargo item added successfully"}

@app.post("/api/retrieve")
async def retrieve_item(
    request: RetrieveRequest,
    items_col: Collection = Depends(get_items_collection),
    logs_col: Collection = Depends(get_logs_collection)
):
    """Handle item retrieval with usage tracking"""
    item = items_col.find_one({"itemId": request.itemId})
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Check usage limits
    if item.get("usageLimit"):
        current_count = item.get("usageCount", 0) + 1
        if current_count >= item["usageLimit"]:
            items_col.update_one(
                {"itemId": request.itemId},
                {"$set": {"isWaste": True, "wasteReason": "Depleted", "usageCount": current_count}}
            )
        else:
            items_col.update_one(
                {"itemId": request.itemId},
                {"$set": {"usageCount": current_count}}
            )
    
    # Increment retrieval count
    items_col.update_one(
        {"itemId": request.itemId},
        {"$inc": {"retrieval_count": 1}}
    )
    
    # Log retrieval action
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "userId": request.userId,
        "actionType": "retrieval",
        "itemId": request.itemId,
        "details": {
            "fromContainer": item.get("containerId", "unknown"),
            "position": item.get("position", {})
        }
    }
    logs_col.insert_one(log_entry)
    
    return {"success": True}

@app.post("/api/place")
async def manual_placement(
    request: PlaceRequest,
    items_col: Collection = Depends(get_items_collection),
    containers_col: Collection = Depends(get_containers_collection),
    logs_col: Collection = Depends(get_logs_collection)
):
    """Allow astronauts to manually place items"""
    item = items_col.find_one({"itemId": request.itemId})
    container = containers_col.find_one({"containerId": request.containerId})
    
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    if not container:
        raise HTTPException(status_code=404, detail="Container not found")
    
    # Update item position
    items_col.update_one(
        {"itemId": request.itemId},
        {"$set": {
            "containerId": request.containerId,
            "position": request.position
        }}
    )
    
    # Log placement
    log_entry = {
        "timestamp": request.timestamp,
        "userId": request.userId,
        "actionType": "placement",
        "itemId": request.itemId,
        "details": {
            "toContainer": request.containerId,
            "position": request.position
        }
    }
    logs_col.insert_one(log_entry)
    
    return {"success": True}

@app.get("/api/waste/identify")
async def identify_waste(items_col: Collection = Depends(get_items_collection)):
    """Automatically identify waste items"""
    current_time = datetime.now()
    
    # Find expired items
    expired = list(items_col.find({
        "expiryDate": {"$lt": current_time.isoformat()},
        "isWaste": False
    }))
    
    # Find depleted items
    depleted = list(items_col.find({
        "usageLimit": {"$exists": True},
        "usageCount": {"$gte": "$usageLimit"},
        "isWaste": False
    }))
    
    # Mark as waste
    for item in expired:
        items_col.update_one(
            {"_id": item["_id"]},
            {"$set": {
                "isWaste": True,
                "wasteReason": "Expired"
            }}
        )
    
    for item in depleted:
        items_col.update_one(
            {"_id": item["_id"]},
            {"$set": {
                "isWaste": True,
                "wasteReason": "Depleted"
            }}
        )
    
    # Format response
    waste_items = []
    for item in expired:
        waste_items.append({
            "itemId": item["itemId"],
            "name": item["name"],
            "reason": "Expired",
            "containerId": item.get("containerId", "unknown"),
            "position": item.get("position", {})
        })
    
    for item in depleted:
        waste_items.append({
            "itemId": item["itemId"],
            "name": item["name"],
            "reason": "Depleted",
            "containerId": item.get("containerId", "unknown"),
            "position": item.get("position", {})
        })
    
    return waste_items

@app.post("/api/waste/complete-undocking")
async def complete_undocking(
    containerId: str, 
    items_col: Collection = Depends(get_items_collection)
):
    """Finalize waste removal after undocking"""
    result = items_col.delete_many({
        "isWaste": True,
        "containerId": containerId
    })
    return {"success": True, "itemsRemoved": result.deleted_count}

@app.post("/api/simulate/day")
async def simulate_time(
    request: SimulationRequest,
    items_col: Collection = Depends(get_items_collection)
):
    """Advance time and update item statuses"""
    current_date = datetime.fromisoformat(get_current_time())
    
    if request.toTimestamp:
        new_date = datetime.fromisoformat(request.toTimestamp)
    else:
        new_date = current_date + timedelta(days=request.numOfDays or 1)
    
    # Process daily expirations
    expired_items = list(items_col.find({
        "expiryDate": {"$lt": new_date.isoformat()},
        "isWaste": False
    }))
    
    for item in expired_items:
        items_col.update_one(
            {"_id": item["_id"]},
            {"$set": {"isWaste": True, "wasteReason": "Expired"}}
        )
    
    # Process usage depletion
    depleted_items = []
    for day in request.itemsToBeUsedPerDay:
        for usage in day.get("usages", []):
            item = items_col.find_one({"itemId": usage.get("itemId")})
            if item:
                new_count = item.get("usageCount", 0) + 1
                items_col.update_one(
                    {"_id": item["_id"]},
                    {"$set": {"usageCount": new_count}}
                )
                
                if item.get("usageLimit") and new_count >= item["usageLimit"]:
                    items_col.update_one(
                        {"_id": item["_id"]},
                        {"$set": {"isWaste": True, "wasteReason": "Depleted"}}
                    )
                    depleted_items.append(item["itemId"])
    
    # Update system time
    set_current_time(new_date.isoformat())
    
    return {
        "success": True,
        "newDate": new_date.isoformat(),
        "expiredItems": [i["itemId"] for i in expired_items],
        "depletedItems": depleted_items
    }

@app.post("/api/import/items")
async def import_items(
    file: UploadFile = File(...),
    items_col: Collection = Depends(get_items_collection)
):
    """Bulk import items from CSV"""
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        items = df.to_dict(orient="records")
        
        # Set default values
        for item in items:
            item["isWaste"] = False
            item["usageCount"] = 0
            item["retrieval_count"] = 0
            item["created_at"] = datetime.now().isoformat()
        
        result = items_col.insert_many(items)
        return {"success": True, "inserted": len(result.inserted_ids)}
    except Exception as e:
        logger.error(f"Import error: {e}")
        raise HTTPException(status_code=400, detail=f"Import failed: {str(e)}")

@app.post("/api/import/containers")
async def import_containers(
    file: UploadFile = File(...),
    containers_col: Collection = Depends(get_containers_collection)
):
    """Bulk import containers from CSV"""
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        containers = df.to_dict(orient="records")
        result = containers_col.insert_many(containers)
        return {"success": True, "inserted": len(result.inserted_ids)}
    except Exception as e:
        logger.error(f"Import error: {e}")
        raise HTTPException(status_code=400, detail=f"Import failed: {str(e)}")

@app.get("/api/export/arrangement")
async def export_arrangement(items_col: Collection = Depends(get_items_collection)):
    """Export current storage layout"""
    try:
        items = list(items_col.find({}, {"_id": 0}))
        df = pd.DataFrame(items)
        
        # Save to a temporary file
        temp_file = "arrangement.csv"
        df.to_csv(temp_file, index=False)
        
        return FileResponse(temp_file, media_type="text/csv", filename="arrangement.csv")
    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.get("/api/logs")
async def get_logs(
    startDate: str,
    endDate: str,
    itemId: Optional[str] = None,
    userId: Optional[str] = None,
    actionType: Optional[str] = None,
    logs_col: Collection = Depends(get_logs_collection)
):
    """Retrieve filtered logs"""
    filter = {
        "timestamp": {
            "$gte": startDate,
            "$lte": endDate
        }
    }
    if itemId: filter["itemId"] = itemId
    if userId: filter["userId"] = userId
    if actionType: filter["actionType"] = actionType
    
    logs = list(logs_col.find(filter).sort("timestamp", DESCENDING))
    
    # Convert ObjectId to string
    for log in logs:
        log["_id"] = str(log["_id"])
    
    return logs

@app.post("/api/placement", response_model=PlacementResponse)
async def placement_recommendation(
    request: PlacementRequest,
    items_col: Collection = Depends(get_items_collection),
    containers_col: Collection = Depends(get_containers_collection)
):
    """Core placement endpoint with 3D bin-packing"""
    # Store containers
    for container in request.containers:
        containers_col.update_one(
            {"containerId": container.containerId},
            {"$set": container.dict()},
            upsert=True
        )

    # Sort items by priority and expiry
    sorted_items = sorted(request.items, 
        key=lambda x: (-x.priority, x.expiryDate or "9999-12-31"))

    placements = []
    rearrangements = []

    for item in sorted_items:
        placed = False
        preferred_containers = [
            c for c in request.containers 
            if c.zone == item.preferredZone
        ]

        # Try preferred containers first
        for container in preferred_containers:
            bin = GuillotineBin(container.width, container.depth, container.height)
            
            # Try all possible rotations
            for rotation in ItemRotation:
                position = bin.insert(item, rotation)
                if position:
                    placement_info = {
                        "itemId": item.itemId,
                        "containerId": container.containerId,
                        "position": {
                            "start": {
                                "width": position["x"],
                                "depth": position["y"],
                                "height": position["z"]
                            },
                            "end": {
                                "width": position["x"] + position["width"],
                                "depth": position["y"] + position["depth"],
                                "height": position["z"] + position["height"]
                            }
                        },
                        "rotation": rotation.value
                    }
                    
                    # Update database with placement
                    items_col.update_one(
                        {"itemId": item.itemId},
                        {"$set": {
                            "containerId": container.containerId,
                            "position": placement_info["position"],
                            "rotation": rotation.value
                        }},
                        upsert=True
                    )
                    
                    placements.append(placement_info)
                    placed = True
                    break
            
            if placed:
                break

        if not placed:
            # Try other containers as fallback
            other_containers = [
                c for c in request.containers 
                if c.zone != item.preferredZone
            ]
            
            for container in other_containers:
                bin = GuillotineBin(container.width, container.depth, container.height)
                
                for rotation in ItemRotation:
                    position = bin.insert(item, rotation)
                    if position:
                        placement_info = {
                            "itemId": item.itemId,
                            "containerId": container.containerId,
                            "position": {
                                "start": {
                                    "width": position["x"],
                                    "depth": position["y"],
                                    "height": position["z"]
                                },
                                "end": {
                                    "width": position["x"] + position["width"],
                                    "depth": position["y"] + position["depth"],
                                    "height": position["z"] + position["height"]
                                }
                            },
                            "rotation": rotation.value
                        }
                        
                        # Update database with placement
                        items_col.update_one(
                            {"itemId": item.itemId},
                            {"$set": {
                                "containerId": container.containerId,
                                "position": placement_info["position"],
                                "rotation": rotation.value
                            }},
                            upsert=True
                        )
                        
                        placements.append(placement_info)
                        
                        # Add rearrangement notice
                        rearrangements.append({
                            "itemId": item.itemId,
                            "action": "relocate",
                            "details": f"Placed in non-preferred zone {container.zone} instead of {item.preferredZone}"
                        })
                        
                        placed = True
                        break
                
                if placed:
                    break
            
            if not placed:
                rearrangements.append({
                    "itemId": item.itemId,
                    "action": "unplaced",
                    "details": "Insufficient space in any container"
                })

    return {"success": True, "placements": placements, "rearrangements": rearrangements}

@app.get("/api/search")
async def search_item(
    itemId: str = Query(...),
    items_col: Collection = Depends(get_items_collection),
    containers_col: Collection = Depends(get_containers_collection)
):
    """Find item and calculate retrieval steps"""
    item = items_col.find_one({"itemId": itemId})
    if not item:
        return {"success": False, "found": False, "error": "Item not found"}

    if not item.get("containerId") or not item.get("position"):
        return {"success": False, "found": True, "error": "Item position unknown"}

    container = containers_col.find_one({"containerId": item["containerId"]})
    if not container:
        return {"success": False, "found": True, "error": "Container not found"}

    # Calculate retrieval steps
    obstruction_query = {
        "containerId": item["containerId"],
        "position.start.depth": {"$lt": item["position"]["end"]["depth"]},
        "position.end.depth": {"$gt": item["position"]["start"]["depth"]},
        "itemId": {"$ne": itemId}
    }
    obstructing = list(items_col.find(obstruction_query).sort("priority", ASCENDING))

    steps = []
    for idx, obs in enumerate(obstructing, 1):
        steps.append({
            "step": idx,
            "action": "remove",
            "itemId": obs["itemId"],
            "itemName": obs.get("name", "Unknown Item")
        })

    steps.append({
        "step": len(steps)+1,
        "action": "retrieve",
        "itemId": itemId,
        "itemName": item.get("name", "Target Item")
    })

    # Add steps to place back removed items
    for idx, obs in enumerate(reversed(obstructing), len(steps)+1):
        steps.append({
            "step": idx+1,
            "action": "placeBack",
            "itemId": obs["itemId"],
            "itemName": obs.get("name", "Unknown Item")
        })

    return {
        "success": True, 
        "found": True, 
        "itemDetails": {
            "name": item.get("name", "Unknown"),
            "containerId": item["containerId"],
            "zone": container.get("zone", "Unknown"),
            "position": item["position"]
        },
        "retrievalSteps": steps
    }

@app.post("/api/waste/return-plan")
async def generate_return_plan(
    maxWeight: float,
    items_col: Collection = Depends(get_items_collection)
):
    """Generate optimal waste return plan using knapsack algorithm"""
    waste_items = list(items_col.find({"isWaste": True}))
    
    if not waste_items:
        return {"success": True, "totalWeight": 0, "totalVolume": 0, "steps": []}
    
    # Convert weight to integer for DP (multiplied by 1000 for precision)
    max_weight_int = int(maxWeight * 1000)
    items_for_knapsack = []
    
    for item in waste_items:
        # Skip items with missing data
        if not all(k in item for k in ["mass", "position"]):
            continue
            
        # Calculate volume
        try:
            volume = (
                (item["position"]["end"]["width"] - item["position"]["start"]["width"]) *
                (item["position"]["end"]["depth"] - item["position"]["start"]["depth"]) *
                (item["position"]["end"]["height"] - item["position"]["start"]["height"])
            )
        except (KeyError, TypeError):
            volume = 0
            
        items_for_knapsack.append({
            "itemId": item["itemId"],
            "name": item.get("name", "Unknown"),
            "mass": item["mass"],
            "mass_int": int(item["mass"] * 1000),
            "volume": volume,
            "containerId": item.get("containerId", "unknown")
        })
    
    # Dynamic programming for the knapsack problem
    dp = [0] * (max_weight_int + 1)
    selected_items = [[] for _ in range(max_weight_int + 1)]
    
    for item in items_for_knapsack:
        mass_int = item["mass_int"]
        
        # Skip items too heavy for the knapsack
        if mass_int > max_weight_int:
            continue
            
        for w in range(max_weight_int, mass_int - 1, -1):
            new_value = dp[w - mass_int] + mass_int
            if new_value > dp[w]:
                dp[w] = new_value
                selected_items[w] = selected_items[w - mass_int] + [item["itemId"]]
    
    # Get the optimal solution
    selected_ids = selected_items[max_weight_int]
    selected = [item for item in items_for_knapsack if item["itemId"] in selected_ids]
    
    # Calculate totals
    total_weight = sum(item["mass"] for item in selected)
    total_volume = sum(item["volume"] for item in selected)
    
    # Generate return steps
    steps = []
    for idx, item in enumerate(selected, 1):
        steps.append({
            "step": idx,
            "itemId": item["itemId"],
            "name": item["name"],
            "mass": item["mass"],
            "containerId": item["containerId"],
            "reason": item["reason"],
            "position": item["position"]
        })
    
    return {
        "success": True,
        "totalWeight": total_weight,
        "totalVolume": total_volume,
        "steps": steps
    }

# ---------------------------- Main Entry Point ----------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

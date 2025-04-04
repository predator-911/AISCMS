# main.py
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
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
MONGO_URI = f"mongodb+srv://{MONGODB_USERNAME}:{MONGO_PASSWORD}@cluster0.38cb2.mongodb.net/{MONGO_DB}?retryWrites=true&w=majority"


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
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client[MONGO_DB]

def get_collection(name: str) -> Collection:
    return db[name]

# Initialize collections
items_col = get_collection("items")
containers_col = get_collection("containers")
logs_col = get_collection("logs")

# Indexes
items_col.create_index([("priority", DESCENDING), ("expiryDate", ASCENDING)])
containers_col.create_index([("zone", ASCENDING)])

# ---------------------------- FastAPI Setup ----------------------------
app = FastAPI(title="Space Cargo Management", version="1.0")

# ---------------------------- Additional Endpoints ----------------------------

# 1. Item Retrieval Endpoint
@app.post("/api/retrieve")
async def retrieve_item(request: RetrieveRequest):
    """Handle item retrieval with usage tracking"""
    item = items_col.find_one({"itemId": request.itemId})
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Check usage limits
    if item.get("usageLimit"):
        if item["usageCount"] >= item["usageLimit"]:
            items_col.update_one(
                {"itemId": request.itemId},
                {"$set": {"isWaste": True, "wasteReason": "Depleted"}}
            )
    
    # Log retrieval action
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "userId": request.userId,
        "actionType": "retrieval",
        "itemId": request.itemId,
        "details": {
            "fromContainer": item["containerId"],
            "position": item["position"]
        }
    }
    logs_col.insert_one(log_entry)
    
    return {"success": True}

# 2. Manual Placement Endpoint
@app.post("/api/place")
async def manual_placement(request: PlaceRequest):
    """Allow astronauts to manually place items"""
    item = items_col.find_one({"itemId": request.itemId})
    container = containers_col.find_one({"containerId": request.containerId})
    
    if not item or not container:
        raise HTTPException(status_code=404, detail="Item/container not found")
    
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

# 3. Waste Identification Endpoint
@app.get("/api/waste/identify", response_model=List[WasteItem])
async def identify_waste():
    """Automatically identify waste items"""
    current_time = datetime.now()
    
    # Find expired items
    expired = list(items_col.find({
        "expiryDate": {"$lt": current_time.isoformat()},
        "isWaste": False
    }))
    
    # Find depleted items
    depleted = list(items_col.find({
        "usageCount": {"$gte": "$usageLimit"},
        "isWaste": False
    }))
    
    # Mark as waste
    for item in expired + depleted:
        items_col.update_one(
            {"_id": item["_id"]},
            {"$set": {
                "isWaste": True,
                "wasteReason": "Expired" if "expiryDate" in item else "Depleted"
            }}
        )
    
    return JSONResponse([item | {"reason": "Expired"} for item in expired] + 
                       [item | {"reason": "Depleted"} for item in depleted])

# 4. Complete Undocking Endpoint
@app.post("/api/waste/complete-undocking")
async def complete_undocking(containerId: str):
    """Finalize waste removal after undocking"""
    result = items_col.delete_many({
        "isWaste": True,
        "containerId": containerId
    })
    return {"success": True, "itemsRemoved": result.deleted_count}

# 5. Time Simulation Endpoint
@app.post("/api/simulate/day")
async def simulate_time(request: SimulationRequest):
    """Advance time and update item statuses"""
    current_date = datetime.fromisoformat(get_current_time())
    
    if request.toTimestamp:
        new_date = datetime.fromisoformat(request.toTimestamp)
    else:
        new_date = current_date + timedelta(days=request.numOfDays)
    
    # Process daily expirations
    expired_items = list(items_col.find({
        "expiryDate": {"$lt": new_date.isoformat()},
        "isWaste": False
    }))
    
    # Process usage depletion
    depleted_items = []
    for day in request.itemsToBeUsedPerDay:
        for usage in day.get("usages", []):
            item = items_col.find_one({"itemId": usage["itemId"]})
            if item:
                new_count = item.get("usageCount", 0) + 1
                items_col.update_one(
                    {"_id": item["_id"]},
                    {"$set": {"usageCount": new_count}}
                )
                if item["usageLimit"] and new_count >= item["usageLimit"]:
                    depleted_items.append(item["itemId"])
    
    # Update system time
    set_current_time(new_date.isoformat())
    
    return {
        "success": True,
        "newDate": new_date.isoformat(),
        "expiredItems": [i["itemId"] for i in expired_items],
        "depletedItems": depleted_items
    }

# 6. Import/Export Endpoints
@app.post("/api/import/items")
async def import_items(file: UploadFile = File(...)):
    """Bulk import items from CSV"""
    df = pd.read_csv(file.file)
    items = df.to_dict(orient="records")
    result = items_col.insert_many(items)
    return {"success": True, "inserted": len(result.inserted_ids)}

@app.post("/api/import/containers")
async def import_containers(file: UploadFile = File(...)):
    """Bulk import containers from CSV"""
    df = pd.read_csv(file.file)
    containers = df.to_dict(orient="records")
    result = containers_col.insert_many(containers)
    return {"success": True, "inserted": len(result.inserted_ids)}

@app.get("/api/export/arrangement")
async def export_arrangement():
    """Export current storage layout"""
    items = list(items_col.find({}, {"_id": 0}))
    df = pd.DataFrame(items)
    df.to_csv("arrangement.csv", index=False)
    return FileResponse("arrangement.csv")

# 7. Logging Endpoint
@app.get("/api/logs", response_model=List[Dict])
async def get_logs(query: LogQuery):
    """Retrieve filtered logs"""
    filter = {
        "timestamp": {
            "$gte": query.startDate,
            "$lte": query.endDate
        }
    }
    if query.itemId: filter["itemId"] = query.itemId
    if query.userId: filter["userId"] = query.userId
    if query.actionType: filter["actionType"] = query.actionType
    
    logs = list(logs_col.find(filter).sort("timestamp", DESCENDING))
    return JSONResponse([{**log, "_id": str(log["_id"])} for log in logs])

@app.post("/api/placement", response_model=PlacementResponse)
async def placement_recommendation(request: PlacementRequest):
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
                    placements.append({
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
                    })
                    placed = True
                    break
            
            if placed:
                break

        if not placed:
            # Trigger rearrangement logic
            # (Implementation omitted for brevity)
            rearrangements.append({
                "itemId": item.itemId,
                "action": "relocate",
                "details": "Insufficient space in preferred zone"
            })

    return {"success": True, "placements": placements, "rearrangements": rearrangements}

@app.get("/api/search")
async def search_item(itemId: str = Query(...)):
    """Find item and calculate retrieval steps"""
    item = items_col.find_one({"itemId": itemId})
    if not item:
        return JSONResponse({"success": False, "found": False}, 404)

    container = containers_col.find_one({"containerId": item["containerId"]})
    if not container:
        return JSONResponse({"success": False, "error": "Container missing"}, 500)

    # Calculate retrieval steps
    obstruction_query = {
        "containerId": item["containerId"],
        "position.end.depth": {"$gt": item["position"]["start"]["depth"]},
        "itemId": {"$ne": itemId}
    }
    obstructing = list(items_col.find(obstruction_query).sort("priority", ASCENDING))

    steps = []
    for idx, obs in enumerate(obstructing, 1):
        steps.append(RetrievalStep(
            step=idx,
            action="remove",
            itemId=obs["itemId"],
            itemName=obs["name"]
        ))

    steps.append(RetrievalStep(
        step=len(steps)+1,
        action="retrieve",
        itemId=itemId,
        itemName=item["name"]
    ))

    return {"success": True, "found": True, "retrievalSteps": steps}

@app.post("/api/waste/return-plan", response_model=ReturnPlanResponse)
async def generate_return_plan(maxWeight: float):
    """Generate optimal waste return plan using knapsack algorithm"""
    waste_items = list(items_col.find({"isWaste": True}))
    
    # Knapsack algorithm to maximize weight under limit
    dp = [0] * (int(maxWeight*1000) + 1)
    selected = []

    for item in sorted(waste_items, key=lambda x: -x["mass"]):
        mass = int(item["mass"] * 1000)
        for w in range(len(dp)-1, mass-1, -1):
            if dp[w - mass] + mass > dp[w]:
                dp[w] = dp[w - mass] + mass
                selected.append(item["itemId"])

    total_weight = sum(item["mass"] for item in waste_items if item["itemId"] in selected)
    total_volume = sum(
        (item["position"]["end"]["width"] - item["position"]["start"]["width"]) *
        (item["position"]["end"]["depth"] - item["position"]["start"]["depth"]) *
        (item["position"]["end"]["height"] - item["position"]["start"]["height"])
        for item in waste_items if item["itemId"] in selected
    )

    return {
        "success": True,
        "totalWeight": total_weight,
        "totalVolume": total_volume,
        "steps": [{"itemId": i} for i in selected]
    }

# ---------------------------- Additional Endpoints ----------------------------
# Implement /api/simulate/day, /api/retrieve, /api/logs etc. following similar patterns

# ---------------------------- Docker Support ----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

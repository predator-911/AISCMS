import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from bson import ObjectId
import uvicorn
from datetime import datetime, timedelta
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from pymongo import MongoClient, ASCENDING
import certifi
import logging
import csv
from io import StringIO

# Configure logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
MONGO_USERNAME = os.getenv("MONGO_USERNAME")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
MONGO_DB = os.getenv("MONGO_DB")

# ✅ Connect to MongoDB Atlas
uri = f"mongodb+srv://{MONGO_USERNAME}:{MONGO_PASSWORD}@cluster0.38cb2.mongodb.net/{MONGO_DB}?retryWrites=true&w=majority"
client = MongoClient(uri, tlsCAFile=certifi.where())
# Collections
db = client[MONGO_DB]
cargo_collection = db["cargo"]
containers_collection = db["containers"]
log_collection = db["logs"]

logger.info("Connected to MongoDB successfully.")


# Initialize FastAPI
app = FastAPI(
    title="Space Cargo Management System",
    description="AI-powered cargo stowage optimization for space stations",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class Position(BaseModel):
    width: float
    depth: float
    height: float

class Coordinates(BaseModel):
    start: Position
    end: Position

class Item(BaseModel):
    itemId: str
    name: str
    width: float
    depth: float
    height: float
    mass: float
    priority: int
    expiryDate: Optional[str] = None
    usageLimit: Optional[int] = None
    preferredZone: Optional[str] = None
    status: str = "active"
    position: Optional[Coordinates] = None
    containerId: Optional[str] = None
    timesUsed: int = 0

class Container(BaseModel):
    containerId: str
    zone: str
    width: float
    depth: float
    height: float
    maxWeight: Optional[float] = None
    currentWeight: float = 0.0

class PlacementRequest(BaseModel):
    items: List[Item]
    containers: List[Container]

class PlacementResponse(BaseModel):
    itemId: str
    containerId: str
    position: Coordinates
    rotation: str

class RetrievalStep(BaseModel):
    step: int
    action: str
    itemId: str
    itemName: str

class WasteItem(BaseModel):
    itemId: str
    name: str
    reason: str
    containerId: str
    position: Coordinates
    mass: float

class ReturnPlanRequest(BaseModel):
    undockingContainerId: str
    undockingDate: str
    maxWeight: float

class SimulationRequest(BaseModel):
    numOfDays: int
    itemsUsedPerDay: List[Dict]

# Helper Functions
def calculate_volume(item: Item):
    return item.width * item.depth * item.height

def calculate_density(item: Item):
    return item.mass / calculate_volume(item)

def days_until_expiry(item: Item):
    if item.expiryDate:
        return (datetime.fromisoformat(item.expiryDate) - datetime.now()).days
    return 365

# Core Algorithms
def pack_items(items: List[Item], container: Container):
    packed_items = []
    remaining_space = [{
        'start': Position(width=0, depth=0, height=0),
        'end': Position(width=container.width, depth=container.depth, height=container.height)
    }]
    
    sorted_items = sorted(items, key=lambda x: (-x.priority, -calculate_volume(x)))
    
    for item in sorted_items:
        rotations = [
            (item.width, item.depth, item.height, 'original'),
            (item.width, item.height, item.depth, 'y-rot'),
            (item.depth, item.width, item.height, 'x-rot'),
            (item.depth, item.height, item.width, 'xy-rot'),
            (item.height, item.width, item.depth, 'z-rot'),
            (item.height, item.depth, item.width, 'zy-rot')
        ]
        
        placed = False
        for i, space in enumerate(remaining_space):
            for w, d, h, rot in rotations:
                if (space['end'].width - space['start'].width >= w and
                    space['end'].depth - space['start'].depth >= d and
                    space['end'].height - space['start'].height >= h):
                    
                    if container.maxWeight and container.currentWeight + item.mass > container.maxWeight:
                        continue
                    
                    position = Coordinates(
                        start=Position(
                            width=space['start'].width,
                            depth=space['start'].depth,
                            height=space['start'].height
                        ),
                        end=Position(
                            width=space['start'].width + w,
                            depth=space['start'].depth + d,
                            height=space['start'].height + h
                        )
                    )
                    
                    placed_item = Item(
                        **item.model_dump(),
                        position=position,
                        containerId=container.containerId
                    )
                    
                    packed_items.append({
                        'item': placed_item,
                        'rotation': rot
                    })
                    
                    container.currentWeight += item.mass
                    
                    new_spaces = []
                    if space['end'].height > position.end.height:
                        new_spaces.append({
                            'start': Position(
                                width=space['start'].width,
                                depth=space['start'].depth,
                                height=position.end.height
                            ),
                            'end': space['end']
                        })
                    
                    if space['end'].width > position.end.width:
                        new_spaces.append({
                            'start': Position(
                                width=position.end.width,
                                depth=space['start'].depth,
                                height=space['start'].height
                            ),
                            'end': Position(
                                width=space['end'].width,
                                depth=space['end'].depth,
                                height=position.end.height
                            )
                        })
                    
                    if space['end'].depth > position.end.depth:
                        new_spaces.append({
                            'start': Position(
                                width=space['start'].width,
                                depth=position.end.depth,
                                height=space['start'].height
                            ),
                            'end': Position(
                                width=position.end.width,
                                depth=space['end'].depth,
                                height=position.end.height
                            )
                        })
                    
                    remaining_space.pop(i)
                    remaining_space.extend(new_spaces)
                    placed = True
                    break
                
            if placed:
                break
    
    return packed_items

def optimize_waste_return(waste_items: List[Item], max_weight: float):
    n = len(waste_items)
    dp = [[0] * (int(max_weight) + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(1, int(max_weight) + 1):
            if waste_items[i-1].mass <= w:
                dp[i][w] = max(dp[i-1][w], 
                              waste_items[i-1].priority + dp[i-1][w - int(waste_items[i-1].mass)])
            else:
                dp[i][w] = dp[i-1][w]
    
    selected = []
    w = int(max_weight)
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected.append(waste_items[i-1])
            w -= int(waste_items[i-1].mass)
    
    return selected

def train_zone_model():
    items = list(cargo_collection.find({}, {
        "width": 1, "depth": 1, "height": 1, 
        "priority": 1, "expiryDate": 1, 
        "preferredZone": 1, "mass": 1
    }))
    
    if len(items) < 10:
        return None
    
    df = pd.DataFrame(items)
    df = df.dropna(subset=['preferredZone'])
    
    df['volume'] = df['width'] * df['depth'] * df['height']
    df['density'] = df['mass'] / df['volume']
    df['days_until_expiry'] = df['expiryDate'].apply(
        lambda x: (datetime.fromisoformat(x) - datetime.now()).days if x else 365
    )
    
    X = df[['priority', 'volume', 'density', 'days_until_expiry']].fillna(365).values
    y = df['preferredZone'].values
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

zone_model = train_zone_model()

# API Endpoints
@app.post("/api/placement", response_model=List[PlacementResponse])
async def placement_recommendation(request: PlacementRequest):
    try:
        # Update containers in DB
        for container in request.containers:
            existing = containers_collection.find_one({"containerId": container.containerId})
            if not existing:
                containers_collection.insert_one(container.model_dump())
        
        placements = []
        for item in request.items:
            # Get zone recommendation
            if not item.preferredZone and zone_model:
                features = np.array([[
                    item.priority,
                    calculate_volume(item),
                    calculate_density(item),
                    days_until_expiry(item)
                ]])
                item.preferredZone = zone_model.predict(features)[0]
            
            # Try preferred zone first
            container = containers_collection.find_one({
                "zone": item.preferredZone,
                "currentWeight": {"$lt": {"$subtract": ["$maxWeight", item.mass]}}
            })
            
            if not container:
                # Fallback to any container with space
                container = containers_collection.find_one({
                    "currentWeight": {"$lt": {"$subtract": ["$maxWeight", item.mass]}}
                })
            
            if container:
                container_obj = Container(**container)
                packed = pack_items([item], container_obj)
                if packed:
                    placed_item = packed[0]['item']
                    cargo_collection.insert_one(placed_item.model_dump())
                    containers_collection.update_one(
                        {"containerId": container_obj.containerId},
                        {"$set": {"currentWeight": container_obj.currentWeight}}
                    )
                    placements.append({
                        "itemId": placed_item.itemId,
                        "containerId": placed_item.containerId,
                        "position": placed_item.position,
                        "rotation": packed[0]['rotation']
                    })
        
        return placements
    
    except Exception as e:
        logger.error(f"Placement error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/search")
async def search_item(
    itemId: Optional[str] = None,
    itemName: Optional[str] = None,
    userId: Optional[str] = None
):
    try:
        if not itemId and not itemName:
            raise HTTPException(status_code=400, detail="Either itemId or itemName required")
        
        query = {}
        if itemId:
            query["itemId"] = itemId
        if itemName:
            query["name"] = {"$regex": itemName, "$options": "i"}
        
        item = cargo_collection.find_one(query)
        if not item:
            return {"success": False, "found": False}
        
        item = Item(**item)
        
        # Find blocking items
        blocking_items = list(cargo_collection.find({
            "containerId": item.containerId,
            "position.start.depth": {"$lt": item.position.end.depth},
            "position.end.depth": {"$gt": item.position.start.depth},
            "position.start.height": {"$lt": item.position.end.height},
            "position.end.height": {"$gt": item.position.start.height},
            "itemId": {"$ne": item.itemId}
        }))
        
        retrieval_steps = []
        for i, block in enumerate(blocking_items):
            retrieval_steps.append({
                "step": i + 1,
                "action": "remove",
                "itemId": block["itemId"],
                "itemName": block["name"]
            })
        
        # Log the search
        log_collection.insert_one({
            "timestamp": datetime.now(),
            "userId": userId,
            "actionType": "search",
            "itemId": item.itemId,
            "details": {
                "containerId": item.containerId,
                "retrievalSteps": len(retrieval_steps)
            }
        })
        
        return {
            "success": True,
            "found": True,
            "item": {
                "itemId": item.itemId,
                "name": item.name,
                "containerId": item.containerId,
                "zone": next((c["zone"] for c in containers_collection.find(
                    {"containerId": item.containerId})), "Unknown"),
                "position": item.position,
                "remainingUses": (item.usageLimit - item.timesUsed) if item.usageLimit else None
            },
            "retrievalSteps": retrieval_steps
        }
    
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/retrieve")
async def retrieve_item(itemId: str, userId: Optional[str] = None):
    try:
        item = cargo_collection.find_one({"itemId": itemId})
        if not item:
            raise HTTPException(status_code=404, detail="Item not found")
        
        item = Item(**item)
        
        # Update usage count
        new_count = item.timesUsed + 1
        cargo_collection.update_one(
            {"itemId": itemId},
            {"$set": {"timesUsed": new_count}}
        )
        
        # Check if item is now depleted
        if item.usageLimit and new_count >= item.usageLimit:
            cargo_collection.update_one(
                {"itemId": itemId},
                {"$set": {"status": "depleted"}}
            )
        
        # Log the retrieval
        log_collection.insert_one({
            "timestamp": datetime.now(),
            "userId": userId,
            "actionType": "retrieval",
            "itemId": itemId,
            "details": {
                "containerId": item.containerId,
                "timesUsed": new_count
            }
        })
        
        return {"success": True}
    
    except Exception as e:
        logger.error(f"Retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/place")
async def place_item(
    itemId: str,
    containerId: str,
    position: Coordinates,
    userId: Optional[str] = None
):
    try:
        # Verify container exists
        container = containers_collection.find_one({"containerId": containerId})
        if not container:
            raise HTTPException(status_code=404, detail="Container not found")
        
        # Update item position
        cargo_collection.update_one(
            {"itemId": itemId},
            {"$set": {
                "containerId": containerId,
                "position": position.model_dump()
            }}
        )
        
        # Log the placement
        log_collection.insert_one({
            "timestamp": datetime.now(),
            "userId": userId,
            "actionType": "placement",
            "itemId": itemId,
            "details": {
                "containerId": containerId,
                "position": position.model_dump()
            }
        })
        
        return {"success": True}
    
    except Exception as e:
        logger.error(f"Placement error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/waste/identify", response_model=List[WasteItem])
async def identify_waste():
    try:
        # Find expired items
        expired_items = list(cargo_collection.find({
            "expiryDate": {"$lt": datetime.now().isoformat()},
            "status": {"$ne": "waste"}
        }))
        
        # Find depleted items
        depleted_items = list(cargo_collection.find({
            "timesUsed": {"$gte": "$usageLimit"},
            "status": {"$ne": "waste"}
        }))
        
        waste_items = []
        for item in expired_items + depleted_items:
            reason = "Expired" if item in expired_items else "Out of Uses"
            waste_items.append({
                "itemId": item["itemId"],
                "name": item["name"],
                "reason": reason,
                "containerId": item["containerId"],
                "position": item["position"],
                "mass": item["mass"]
            })
        
        return waste_items
    
    except Exception as e:
        logger.error(f"Waste identification error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/waste/return-plan", response_model=Dict)
async def waste_return_plan(request: ReturnPlanRequest):
    try:
        waste_items = list(cargo_collection.find({
            "status": {"$in": ["expired", "depleted"]}
        }))
        
        if not waste_items:
            return {"success": True, "message": "No waste items found"}
        
        # Convert to Item objects
        waste_objects = [Item(**item) for item in waste_items]
        
        # Optimize waste selection
        selected_waste = optimize_waste_return(waste_objects, request.maxWeight)
        
        # Generate movement steps
        steps = []
        for i, item in enumerate(selected_waste):
            steps.append({
                "step": i + 1,
                "action": "move",
                "itemId": item.itemId,
                "itemName": item.name,
                "fromContainer": item.containerId,
                "toContainer": request.undockingContainerId
            })
        
        # Generate manifest
        total_weight = sum(item.mass for item in selected_waste)
        total_volume = sum(calculate_volume(item) for item in selected_waste)
        
        manifest = {
            "undockingContainerId": request.undockingContainerId,
            "undockingDate": request.undockingDate,
            "returnItems": [{
                "itemId": item.itemId,
                "name": item.name,
                "reason": "Expired" if item.status == "expired" else "Out of Uses"
            } for item in selected_waste],
            "totalVolume": total_volume,
            "totalWeight": total_weight
        }
        
        return {
            "success": True,
            "returnPlan": steps,
            "manifest": manifest
        }
    
    except Exception as e:
        logger.error(f"Return plan error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simulate/day", response_model=Dict)
async def simulate_day(request: SimulationRequest):
    try:
        simulated_date = datetime.now() + timedelta(days=request.numOfDays)
        changes = {
            "itemsUsed": [],
            "itemsExpired": [],
            "itemsDepletedToday": []
        }
        
        # Process items used per day
        for day in range(request.numOfDays):
            current_date = datetime.now() + timedelta(days=day)
            
            # Mark expired items
            expired = cargo_collection.update_many(
                {
                    "expiryDate": {"$lt": current_date.isoformat()},
                    "status": "active"
                },
                {"$set": {"status": "expired"}}
            )
            
            if expired.modified_count > 0:
                changes["itemsExpired"].extend([
                    {"itemId": item["itemId"], "name": item["name"]}
                    for item in cargo_collection.find({
                        "expiryDate": {"$lt": current_date.isoformat()},
                        "status": "expired"
                    })
                ])
            
            # Process used items
            for item_used in request.itemsUsedPerDay:
                item = cargo_collection.find_one({
                    "$or": [
                        {"itemId": item_used.get("itemId")},
                        {"name": item_used.get("name")}
                    ]
                })
                
                if item:
                    new_count = item.get("timesUsed", 0) + 1
                    update = {"$set": {"timesUsed": new_count}}
                    
                    if item.get("usageLimit") and new_count >= item["usageLimit"]:
                        update["$set"]["status"] = "depleted"
                        changes["itemsDepletedToday"].append({
                            "itemId": item["itemId"],
                            "name": item["name"]
                        })
                    
                    cargo_collection.update_one(
                        {"_id": item["_id"]},
                        update
                    )
                    
                    changes["itemsUsed"].append({
                        "itemId": item["itemId"],
                        "name": item["name"],
                        "remainingUses": (item["usageLimit"] - new_count) if item.get("usageLimit") else None
                    })
        
        return {
            "success": True,
            "newDate": simulated_date.isoformat(),
            "changes": changes
        }
    
    except Exception as e:
        logger.error(f"Simulation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/import/items")
async def import_items(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        decoded = contents.decode('utf-8')
        df = pd.read_csv(StringIO(decoded))
        
        # Convert to dictionary and clean data
        items = []
        for _, row in df.iterrows():
            item = {
                "itemId": str(row.get("Item ID", "")),
                "name": str(row.get("Name", "")),
                "width": float(row.get("Width (cm)", 0)),
                "depth": float(row.get("Depth (cm)", 0)),
                "height": float(row.get("Height (cm)", 0)),
                "mass": float(row.get("Mass (kg)", 0)),
                "priority": int(row.get("Priority (1-100)", 1)),
                "expiryDate": row.get("Expiry Date (ISO Format)", None),
                "usageLimit": int(row.get("Usage Limit", 0)) if pd.notna(row.get("Usage Limit")) else None,
                "preferredZone": row.get("Preferred Zone", None),
                "status": "active",
                "timesUsed": 0
            }
            items.append(item)
        
        # Insert into database
        result = cargo_collection.insert_many(items)
        
        # Retrain AI model
        global zone_model
        zone_model = train_zone_model()
        
        return {
            "success": True,
            "itemsImported": len(result.inserted_ids)
        }
    
    except Exception as e:
        logger.error(f"Import error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/export/arrangement")
async def export_arrangement():
    try:
        items = list(cargo_collection.find({}))
        
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Item ID", "Container ID", 
            "Start Width", "Start Depth", "Start Height",
            "End Width", "End Depth", "End Height"
        ])
        
        for item in items:
            if "position" in item and item["position"]:
                pos = item["position"]
                writer.writerow([
                    item["itemId"],
                    item.get("containerId", ""),
                    pos["start"]["width"],
                    pos["start"]["depth"],
                    pos["start"]["height"],
                    pos["end"]["width"],
                    pos["end"]["depth"],
                    pos["end"]["height"]
                ])
        
        output.seek(0)
        return FileResponse(
            output,
            media_type="text/csv",
            filename="cargo_arrangement.csv"
        )
    
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/logs")
async def get_logs(
    startDate: Optional[str] = None,
    endDate: Optional[str] = None,
    itemId: Optional[str] = None,
    userId: Optional[str] = None,
    actionType: Optional[str] = None
):
    try:
        query = {}
        
        if startDate and endDate:
            query["timestamp"] = {
                "$gte": datetime.fromisoformat(startDate),
                "$lte": datetime.fromisoformat(endDate)
            }
        elif startDate:
            query["timestamp"] = {"$gte": datetime.fromisoformat(startDate)}
        elif endDate:
            query["timestamp"] = {"$lte": datetime.fromisoformat(endDate)}
        
        if itemId:
            query["itemId"] = itemId
        
        if userId:
            query["userId"] = userId
        
        if actionType:
            query["actionType"] = actionType
        
        logs = list(log_collection.find(query).sort("timestamp", -1).limit(100))
        
        return {
            "logs": [{
                "timestamp": log["timestamp"].isoformat(),
                "userId": log.get("userId"),
                "actionType": log["actionType"],
                "itemId": log["itemId"],
                "details": log.get("details", {})
            } for log in logs]
        }
    
    except Exception as e:
        logger.error(f"Log retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

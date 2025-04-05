import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Depends, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
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
MONGODB_USERNAME = os.getenv("MONGODB_USERNAME", "user")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "password")
MONGO_DB = os.getenv("MONGO_DB", "space_cargo")
MONGO_URI = os.getenv("MONGO_URI", f"mongodb+srv://{MONGODB_USERNAME}:{MONGO_PASSWORD}@cluster0.38cb2.mongodb.net/{MONGO_DB}?retryWrites=true&w=majority")

# Time management
CURRENT_TIME = datetime.now().isoformat()

def get_current_time() -> str:
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
    expiryDate: Optional[str] = None
    usageLimit: Optional[int] = None
    preferredZone: str

class Container(BaseModel):
    containerId: str
    zone: str = "General"
    width: float = Field(..., gt=0)
    depth: float = Field(..., gt=0)
    height: float = Field(..., gt=0)

    @validator('*', pre=True)
    def replace_nan(cls, value):
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return 0.0
        return value

class PlacementRequest(BaseModel):
    items: List[Item]
    containers: List[Container]

class PlacementResponse(BaseModel):
    success: bool
    placements: List[Dict]
    rearrangements: List[Dict]

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
        x, y, z, w, d, h = rect
        remaining = []
        if w > item_width:
            remaining.append((x + item_width, y, z, w - item_width, d, h))
        if d > item_depth:
            remaining.append((x, y + item_depth, z, w, d - item_depth, h))
        if h > item_height:
            remaining.append((x, y, z + item_height, w, d, h - item_height))
        return remaining

    def insert(self, item: Item, rotation: ItemRotation) -> Optional[Dict]:
        rotations = {
            ItemRotation.WDH: (item.width, item.depth, item.height),
            ItemRotation.WHD: (item.width, item.height, item.depth),
            ItemRotation.DWH: (item.depth, item.width, item.height),
            ItemRotation.DHW: (item.depth, item.height, item.width),
            ItemRotation.HWD: (item.height, item.width, item.depth),
            ItemRotation.HDW: (item.height, item.depth, item.width)
        }
        iw, id_, ih = rotations[rotation]

        best_score = float('inf')
        best_rect = None
        best_idx = -1

        for idx, rect in enumerate(self.free_rects):
            x, y, z, w, d, h = rect
            if w >= iw and d >= id_ and h >= ih:
                score = y + id_
                if score < best_score:
                    best_score = score
                    best_rect = rect
                    best_idx = idx

        if not best_rect:
            return None

        placement = {
            "x": best_rect[0],
            "y": best_rect[1],
            "z": best_rect[2],
            "width": iw,
            "depth": id_,
            "height": ih,
            "rotation": rotation
        }

        del self.free_rects[best_idx]
        self.free_rects.extend(self.split_rect(best_rect, iw, id_, ih))
        self.placements.append(placement)
        return placement

# ---------------------------- Database Setup ----------------------------
def get_db():
    client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    try:
        return client[MONGO_DB]
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")

def get_items_collection(db=Depends(get_db)):
    return db["items"]

def get_containers_collection(db=Depends(get_db)):
    return db["containers"]

def get_logs_collection(db=Depends(get_db)):
    return db["logs"]

# ---------------------------- FastAPI Setup ----------------------------
app = FastAPI(title="Space Cargo Management", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "X-Total-Count"]
)

@app.on_event("startup")
async def startup_db_client():
    try:
        db = get_db()
        db["items"].create_index([("priority", DESCENDING), ("expiryDate", ASCENDING)])
        db["containers"].create_index([("zone", ASCENDING)])
        db["logs"].create_index([("timestamp", DESCENDING)])
        logger.info("Database indexes created")
    except Exception as e:
        logger.error(f"Index creation failed: {e}")

# ---------------------------- API Endpoints ----------------------------
@app.get("/api/items")
async def get_all_items(items_col: Collection = Depends(get_items_collection)):
    try:
        items = list(items_col.find({}, {"_id": 0}))
        for item in items:
            for field in ['width', 'depth', 'height', 'mass']:
                val = item.get(field, 0)
                if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                    item[field] = 0.0
        return JSONResponse(content={"success": True, "items": items, "count": len(items)})
    except Exception as e:
        logger.error(f"Items retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve items")

@app.get("/api/containers")
async def get_all_containers(containers_col: Collection = Depends(get_containers_collection)):
    try:
        containers = list(containers_col.find({}, {"_id": 0}))
        return {"success": True, "containers": containers, "count": len(containers)}
    except Exception as e:
        logger.error(f"Containers retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve containers")

@app.post("/api/add_cargo")
async def add_cargo(item: Item, items_col: Collection = Depends(get_items_collection)):
    try:
        item_dict = item.dict()
        item_dict.update({
            "expiryDate": (datetime.now() + timedelta(days=30)).isoformat(),
            "usageCount": 0,
            "retrieval_count": 0,
            "created_at": datetime.now().isoformat(),
            "isWaste": False
        })
        items_col.insert_one(item_dict)
        return {"success": True, "message": "Item added successfully"}
    except Exception as e:
        logger.error(f"Item addition failed: {e}")
        raise HTTPException(status_code=400, detail="Failed to add item")

@app.get("/api/waste/identify")
async def identify_waste(items_col: Collection = Depends(get_items_collection)):
    try:
        current_time = datetime.now()
        expired = list(items_col.find({
            "expiryDate": {"$lt": current_time.isoformat()},
            "isWaste": False
        }))
        depleted = list(items_col.find({
            "usageLimit": {"$exists": True, "$ne": None},
            "$expr": {"$gte": ["$usageCount", "$usageLimit"]},
            "isWaste": False
        }))
        
        updates = []
        for item in expired + depleted:
            updates.append({
                "filter": {"_id": item["_id"]},
                "update": {"$set": {
                    "isWaste": True,
                    "wasteReason": "Expired" if item in expired else "Depleted"
                }}
            })
        
        if updates:
            items_col.bulk_write([UpdateOne(**u) for u in updates])
        
        waste_items = [{
            "itemId": i["itemId"],
            "name": i.get("name", "Unknown"),
            "reason": "Expired" if i in expired else "Depleted",
            "containerId": i.get("containerId", "unknown"),
            "position": i.get("position", {})
        } for i in expired + depleted]
        
        return waste_items
    except Exception as e:
        logger.error(f"Waste identification failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to identify waste")

@app.post("/api/import/items")
async def import_items(file: UploadFile = File(...), items_col: Collection = Depends(get_items_collection)):
    try:
        df = pd.read_csv(StringIO((await file.read()).decode('utf-8')))
        if 'itemId' not in df.columns:
            raise ValueError("Missing itemId column")
        
        numeric_cols = ['width', 'depth', 'height', 'mass', 'priority']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        items = df.to_dict(orient="records")
        for item in items:
            item.update({
                "isWaste": False,
                "usageCount": 0,
                "retrieval_count": 0,
                "created_at": datetime.now().isoformat()
            })
        
        result = items_col.insert_many(items)
        return {"success": True, "inserted": len(result.inserted_ids)}
    except Exception as e:
        logger.error(f"Import failed: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid data: {str(e)}")

@app.post("/api/waste/return-plan")
async def generate_return_plan(maxWeight: float, items_col: Collection = Depends(get_items_collection)):
    try:
        waste_items = list(items_col.find({"isWaste": True}))
        valid_items = []
        
        for item in waste_items:
            try:
                mass = float(item.get("mass", 0))
                pos = item.get("position", {})
                start = pos.get("start", {})
                end = pos.get("end", {})
                
                width = float(end.get("width", 0)) - float(start.get("width", 0))
                depth = float(end.get("depth", 0)) - float(start.get("depth", 0))
                height = float(end.get("height", 0)) - float(start.get("height", 0))
                
                valid_items.append({
                    "itemId": item["itemId"],
                    "name": item.get("name", "Unknown"),
                    "mass": mass,
                    "volume": abs(width * depth * height),
                    "containerId": item.get("containerId", "unknown"),
                    "reason": item.get("wasteReason", "Unknown")
                })
            except Exception as e:
                logger.warning(f"Skipping invalid item: {e}")
                continue

        max_weight_int = int(maxWeight * 1000)
        dp = [0] * (max_weight_int + 1)
        selected_items = [[] for _ in range(max_weight_int + 1)]
        
        for item in valid_items:
            mass_int = int(item["mass"] * 1000)
            for w in range(max_weight_int, mass_int - 1, -1):
                if dp[w - mass_int] + item["volume"] > dp[w]:
                    dp[w] = dp[w - mass_int] + item["volume"]
                    selected_items[w] = selected_items[w - mass_int] + [item]
        
        best_solution = selected_items[max_weight_int]
        total_weight = sum(i["mass"] for i in best_solution)
        total_volume = sum(i["volume"] for i in best_solution)
        
        steps = [{
            "step": idx,
            "action": "remove_waste",
            "itemId": i["itemId"],
            "itemName": i["name"],
            "containerId": i["containerId"],
            "reason": i["reason"]
        } for idx, i in enumerate(best_solution, 1)]
        
        return {"success": True, "totalWeight": total_weight, "totalVolume": total_volume, "steps": steps}
    except Exception as e:
        logger.error(f"Return plan failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate return plan")

# ... [Other endpoints with similar error handling and validation]

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

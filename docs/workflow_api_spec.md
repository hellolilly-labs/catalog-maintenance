# Workflow State Management API Specification
## Future API Integration for Brand Pipeline Management

This document describes the workflow state management system structure for future API development.

---

## 📋 **Core Data Structures**

### **WorkflowState Enum**
```typescript
enum WorkflowState {
  NOT_STARTED = "not_started",
  RESEARCH_IN_PROGRESS = "research_in_progress", 
  RESEARCH_COMPLETE = "research_complete",
  CATALOG_IN_PROGRESS = "catalog_in_progress",
  CATALOG_COMPLETE = "catalog_complete", 
  KNOWLEDGE_IN_PROGRESS = "knowledge_in_progress",
  KNOWLEDGE_COMPLETE = "knowledge_complete",
  RAG_IN_PROGRESS = "rag_in_progress",
  RAG_COMPLETE = "rag_complete",
  PERSONA_IN_PROGRESS = "persona_in_progress", 
  PERSONA_COMPLETE = "persona_complete",
  PIPELINE_COMPLETE = "pipeline_complete",
  MAINTENANCE_MODE = "maintenance_mode",
  ERROR_STATE = "error_state",
  PARTIAL_FAILURE = "partial_failure"
}
```

### **NextStepPriority Enum**
```typescript
enum NextStepPriority {
  CRITICAL = "critical",      // Must be done immediately
  HIGH = "high",             // Should be done soon  
  MEDIUM = "medium",         // Normal priority
  LOW = "low",              // Can be deferred
  MAINTENANCE = "maintenance" // Regular maintenance
}
```

### **NextStep Interface**
```typescript
interface NextStep {
  action: string;                    // Human-readable action description
  command: string;                   // Specific CLI command to execute
  priority: NextStepPriority;        // Priority level
  estimated_duration: string;        // Estimated time to complete
  estimated_cost: string;           // Estimated API cost
  prerequisites: string[];          // Any prerequisites needed
  reason: string;                   // Why this step is needed
  automation_ready: boolean;        // Can this be automated?
}
```

### **WorkflowProgress Interface**
```typescript
interface WorkflowProgress {
  brand_url: string;
  current_state: WorkflowState;
  last_updated: string;              // ISO datetime
  next_step: NextStep | null;
  
  // Component completion status
  research_phases: Record<string, string>;  // phase_name -> status
  catalog_status: string;
  knowledge_status: string;
  rag_status: string;
  persona_status: string;
  
  // Progress tracking
  total_progress_percent: number;
  estimated_completion: string | null;      // ISO datetime
  
  // Error tracking
  errors: Array<{
    timestamp: string;
    step: string;
    error: string;
  }>;
  warnings: Array<{
    timestamp: string;
    step: string;
    warning: string;
  }>;
  
  // Workflow metadata
  onboarding_started: string | null;        // ISO datetime
  last_successful_step: string | null;
  step_history: Array<{
    timestamp: string;
    from_state: string;
    to_state: string;
    step_completed: string | null;
    error: string | null;
  }>;
}
```

---

## 🔌 **Future API Endpoints**

### **Get Workflow State**
```
GET /api/v1/brands/{brand_url}/workflow
```

**Response:**
```json
{
  "success": true,
  "data": {
    "brand_url": "specialized.com",
    "current_state": "catalog_complete",
    "last_updated": "2024-12-20T14:30:00Z",
    "total_progress_percent": 60.0,
    "next_step": {
      "action": "Create knowledge base",
      "command": "python src/knowledge_ingestor.py --brand specialized.com --include-brand-intelligence",
      "priority": "high",
      "estimated_duration": "5-10 minutes",
      "estimated_cost": "$1-3",
      "prerequisites": ["Product catalog complete"],
      "reason": "Product catalog is ready, create knowledge base",
      "automation_ready": true
    },
    "research_phases": {
      "foundation": "complete",
      "market_positioning": "complete",
      "product_style": "complete",
      "customer_cultural": "complete",
      "voice_messaging": "complete",
      "rag_optimization": "not_started"
    },
    "catalog_status": "complete",
    "knowledge_status": "not_started",
    "rag_status": "not_started",
    "persona_status": "not_started",
    "estimated_completion": "2024-12-20T15:00:00Z",
    "errors": [],
    "warnings": [],
    "onboarding_started": "2024-12-20T14:00:00Z",
    "last_successful_step": "catalog_ingestion",
    "step_history": [
      {
        "timestamp": "2024-12-20T14:00:00Z",
        "from_state": "not_started",
        "to_state": "research_in_progress",
        "step_completed": "started_research",
        "error": null
      },
      {
        "timestamp": "2024-12-20T14:15:00Z",
        "from_state": "research_in_progress", 
        "to_state": "research_complete",
        "step_completed": "brand_research",
        "error": null
      },
      {
        "timestamp": "2024-12-20T14:30:00Z",
        "from_state": "research_complete",
        "to_state": "catalog_complete", 
        "step_completed": "catalog_ingestion",
        "error": null
      }
    ]
  }
}
```

### **Update Workflow State**
```
POST /api/v1/brands/{brand_url}/workflow/update
```

**Request Body:**
```json
{
  "new_state": "knowledge_complete",
  "step_completed": "knowledge_base_creation",
  "error": null
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "brand_url": "specialized.com",
    "current_state": "knowledge_complete",
    "next_step": {
      "action": "Optimize RAG search",
      "command": "python src/research/brand_researcher.py --brand specialized.com --phases rag_optimization",
      "priority": "medium",
      "estimated_duration": "5-10 minutes",
      "estimated_cost": "$2-4"
    },
    "total_progress_percent": 80.0
  }
}
```

### **Get Next Step Command**
```
GET /api/v1/brands/{brand_url}/workflow/next-step
```

**Response:**
```json
{
  "success": true,
  "data": {
    "command": "python src/knowledge_ingestor.py --brand specialized.com --include-brand-intelligence",
    "action": "Create knowledge base",
    "priority": "high",
    "estimated_duration": "5-10 minutes",
    "estimated_cost": "$1-3",
    "automation_ready": true
  }
}
```

### **Execute Next Step** 
```
POST /api/v1/brands/{brand_url}/workflow/execute-next-step
```

**Response:**
```json
{
  "success": true,
  "data": {
    "execution_id": "exec_123456789",
    "status": "running",
    "command": "python src/knowledge_ingestor.py --brand specialized.com --include-brand-intelligence",
    "started_at": "2024-12-20T15:00:00Z",
    "estimated_completion": "2024-12-20T15:10:00Z"
  }
}
```

### **List All Brand States**
```
GET /api/v1/brands/workflow/states
```

**Response:**
```json
{
  "success": true,
  "data": {
    "specialized.com": {
      "current_state": "catalog_complete",
      "total_progress_percent": 60.0,
      "next_step": {
        "action": "Create knowledge base",
        "priority": "high"
      }
    },
    "nike.com": {
      "current_state": "pipeline_complete", 
      "total_progress_percent": 100.0,
      "next_step": {
        "action": "Smart maintenance check",
        "priority": "maintenance"
      }
    }
  }
}
```

### **Batch Next Steps**
```
GET /api/v1/brands/workflow/next-steps
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "brand_url": "specialized.com",
      "current_state": "catalog_complete",
      "progress_percent": 60.0,
      "next_step": {
        "action": "Create knowledge base",
        "priority": "high",
        "estimated_duration": "5-10 minutes",
        "estimated_cost": "$1-3"
      }
    },
    {
      "brand_url": "nike.com", 
      "current_state": "pipeline_complete",
      "progress_percent": 100.0,
      "next_step": {
        "action": "Smart maintenance check",
        "priority": "maintenance",
        "estimated_duration": "2-10 minutes", 
        "estimated_cost": "$1-4"
      }
    }
  ]
}
```

---

## 🔄 **State Transition Rules**

### **Valid State Transitions**
```
NOT_STARTED → RESEARCH_IN_PROGRESS
RESEARCH_IN_PROGRESS → RESEARCH_COMPLETE | ERROR_STATE
RESEARCH_COMPLETE → CATALOG_IN_PROGRESS
CATALOG_IN_PROGRESS → CATALOG_COMPLETE | ERROR_STATE  
CATALOG_COMPLETE → KNOWLEDGE_IN_PROGRESS
KNOWLEDGE_IN_PROGRESS → KNOWLEDGE_COMPLETE | ERROR_STATE
KNOWLEDGE_COMPLETE → RAG_IN_PROGRESS
RAG_IN_PROGRESS → RAG_COMPLETE | ERROR_STATE
RAG_COMPLETE → PERSONA_IN_PROGRESS
PERSONA_IN_PROGRESS → PERSONA_COMPLETE | ERROR_STATE
PERSONA_COMPLETE → PIPELINE_COMPLETE
PIPELINE_COMPLETE → MAINTENANCE_MODE
ERROR_STATE → [Previous successful state] | NOT_STARTED
PARTIAL_FAILURE → [Appropriate recovery state]
```

### **Progress Calculation**
```typescript
function calculateProgress(state: WorkflowState): number {
  const stateWeights = {
    NOT_STARTED: 0,
    RESEARCH_IN_PROGRESS: 10,
    RESEARCH_COMPLETE: 20,
    CATALOG_IN_PROGRESS: 30, 
    CATALOG_COMPLETE: 40,
    KNOWLEDGE_IN_PROGRESS: 50,
    KNOWLEDGE_COMPLETE: 60,
    RAG_IN_PROGRESS: 70,
    RAG_COMPLETE: 80,
    PERSONA_IN_PROGRESS: 90,
    PERSONA_COMPLETE: 95,
    PIPELINE_COMPLETE: 100,
    MAINTENANCE_MODE: 100
  };
  
  return stateWeights[state] || 0;
}
```

---

## 🚀 **Integration Examples**

### **JavaScript/Node.js Client**
```typescript
class WorkflowClient {
  async getWorkflowState(brandUrl: string): Promise<WorkflowProgress> {
    const response = await fetch(`/api/v1/brands/${brandUrl}/workflow`);
    const result = await response.json();
    return result.data;
  }
  
  async executeNextStep(brandUrl: string): Promise<ExecutionResult> {
    const response = await fetch(`/api/v1/brands/${brandUrl}/workflow/execute-next-step`, {
      method: 'POST'
    });
    const result = await response.json();
    return result.data;
  }
  
  async getBatchNextSteps(): Promise<NextStepSummary[]> {
    const response = await fetch('/api/v1/brands/workflow/next-steps');
    const result = await response.json();
    return result.data;
  }
}
```

### **Python Client**
```python
import asyncio
import aiohttp
from typing import Dict, List, Optional

class WorkflowClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    async def get_workflow_state(self, brand_url: str) -> Dict:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/api/v1/brands/{brand_url}/workflow") as response:
                result = await response.json()
                return result['data']
    
    async def execute_next_step(self, brand_url: str) -> Dict:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/api/v1/brands/{brand_url}/workflow/execute-next-step") as response:
                result = await response.json()
                return result['data']
    
    async def get_batch_next_steps(self) -> List[Dict]:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/api/v1/brands/workflow/next-steps") as response:
                result = await response.json()
                return result['data']
```

---

This API specification provides a complete foundation for building REST APIs, GraphQL APIs, or WebSocket APIs on top of the existing workflow state management system. The data structures are already implemented in the Python backend and can be easily exposed via any API framework. 🎯 
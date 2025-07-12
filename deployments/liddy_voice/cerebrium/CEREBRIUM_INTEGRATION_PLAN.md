# Cerebrium Integration Plan for Liddy Voice Agent

## Executive Summary

This document outlines the plan for deploying Liddy Voice Agent on Cerebrium's serverless GPU infrastructure while maintaining connectivity to existing GCP resources (Redis, Cloud Storage, etc.).

## Current Architecture

### GCP Infrastructure
- **Redis**: Google Cloud Memorystore at `10.210.164.91` (private IP)
- **VPC Connector**: `redis-connector` for Cloud Run to access Redis
- **Storage**: GCS buckets (liddy-conversations, liddy-account-documents)
- **Security**: VPC with private-ranges-only egress
- **Secrets**: Managed via GCP Secret Manager

### Cerebrium Capabilities
- Serverless GPU infrastructure (NVIDIA A10G)
- Pay-per-second billing
- Auto-scaling (0-10 replicas)
- WebSocket and REST API support
- No direct VPC peering with GCP

## Challenge: Cross-Cloud Connectivity

Cerebrium cannot directly access GCP private resources because:
1. Redis instance has only private IP (10.210.164.91)
2. VPC connector is GCP-specific
3. No VPC peering between Cerebrium and GCP
4. GCS requires authentication

## Proposed Solutions

### Solution 1: API Gateway Pattern (Recommended)

Create a lightweight API gateway in GCP that Cerebrium can call.

**Architecture:**
```
Cerebrium Voice Agent → HTTPS → GCP API Gateway → Redis/GCS
```

**Implementation:**
1. New Cloud Run service: `liddy-redis-gateway`
2. RESTful endpoints:
   - `/redis/get/{key}`
   - `/redis/set`
   - `/redis/delete/{key}`
   - `/storage/read/{path}`
   - `/storage/write`
3. Authentication via API keys
4. Rate limiting and monitoring

**Pros:**
- Maintains existing Redis/GCS setup
- Strong security boundary
- Easy to monitor and control access
- Can add caching layer

**Cons:**
- Additional latency (cross-cloud calls)
- Another service to maintain
- Potential bottleneck

### Solution 2: Public Redis + Enhanced Security

Use publicly accessible Redis with strong security measures.

**Options:**
1. **Upstash Redis**: Serverless Redis with global endpoints
2. **Redis Cloud**: Managed Redis with public endpoints
3. **Self-hosted**: Redis on GCE with public IP

**Implementation:**
1. Migrate data from Memorystore to public Redis
2. Configure strong passwords and TLS
3. IP allowlisting for Cerebrium
4. Regular key rotation

**Pros:**
- Direct access from Cerebrium
- Lower latency than API gateway
- Serverless options available

**Cons:**
- Security concerns with public endpoint
- Migration effort required
- Potential cost increase

### Solution 3: Hybrid Deployment

Keep compute-intensive tasks on Cerebrium, session management on GCP.

**Architecture:**
```
User → LiveKit → GCP Cloud Run (Gateway) → Cerebrium (ML Processing)
                           ↓
                        Redis/GCS
```

**Implementation:**
1. GCP handles WebSocket connections and session state
2. Cerebrium handles ML inference (STT/TTS/LLM)
3. Communication via async job queue

**Pros:**
- Best of both worlds
- No cross-cloud data access needed
- Can optimize costs

**Cons:**
- More complex architecture
- Potential latency for real-time voice
- Two deployments to manage

## Recommended Approach

**Phase 1: API Gateway (Quick Win)**
- Implement minimal API gateway for Redis/storage access
- Deploy voice agent to Cerebrium
- Test performance and costs

**Phase 2: Optimize Based on Results**
- If latency is acceptable: Keep API gateway
- If latency is high: Move to Upstash Redis
- If costs are high: Consider hybrid deployment

## Implementation Roadmap

### Week 1: API Gateway Development
- [ ] Create `liddy-redis-gateway` service
- [ ] Implement Redis proxy endpoints
- [ ] Implement GCS proxy endpoints
- [ ] Add authentication and rate limiting

### Week 2: Cerebrium Integration
- [ ] Update Redis client to use HTTP API
- [ ] Add retry logic for cross-cloud calls
- [ ] Configure Cerebrium secrets
- [ ] Deploy and test basic functionality

### Week 3: Performance Optimization
- [ ] Add caching layer to API gateway
- [ ] Implement connection pooling
- [ ] Optimize request batching
- [ ] Load testing and benchmarking

### Week 4: Production Readiness
- [ ] Set up monitoring and alerting
- [ ] Implement failover strategies
- [ ] Security audit
- [ ] Documentation and runbooks

## Security Considerations

1. **API Gateway Security**
   - Use strong API keys (rotate monthly)
   - Implement request signing
   - Rate limiting per client
   - Audit logging

2. **Network Security**
   - Allowlist Cerebrium IPs
   - Use TLS for all communications
   - Monitor for anomalies

3. **Data Security**
   - Encrypt sensitive data in transit
   - Minimize data in cross-cloud calls
   - Regular security scans

## Cost Analysis

### Current GCP Costs (Monthly Estimate)
- Cloud Run: ~$50-100
- Redis (Memorystore): ~$50
- Storage: ~$20
- **Total**: ~$120-170

### Cerebrium Costs (Monthly Estimate)
- GPU compute: ~$0.0028/second * usage
- Assuming 1000 hours: ~$100
- Storage: Included
- **Total**: ~$100

### Additional Costs
- API Gateway (Cloud Run): ~$20
- Upstash Redis (if used): ~$20-50
- Cross-cloud data transfer: ~$10-20

### Total Estimated Cost
- **With API Gateway**: ~$250-300/month
- **With Upstash**: ~$240-340/month
- **Savings from scale-to-zero**: Significant during low usage

## Monitoring and Observability

1. **Metrics to Track**
   - Cross-cloud latency
   - API gateway response times
   - Redis hit/miss rates
   - GPU utilization on Cerebrium
   - Cost per conversation

2. **Alerting**
   - High latency (>500ms for Redis)
   - Failed cross-cloud calls
   - High GPU memory usage
   - Unusual traffic patterns

## Rollback Plan

If Cerebrium deployment has issues:
1. Keep GCP deployment running in parallel
2. Use feature flags to route traffic
3. Quick switch back via DNS/load balancer
4. No data migration needed (shared Redis)

## Open Questions

1. What are Cerebrium's specific IP ranges for allowlisting?
2. Can we get dedicated egress IPs from Cerebrium?
3. What's the actual latency for US-Central to Cerebrium regions?
4. Are there Cerebrium regions closer to our GCP region?

## Next Steps

1. Get Cerebrium account and API access
2. Implement proof-of-concept API gateway
3. Benchmark cross-cloud latency
4. Make final architecture decision
5. Proceed with full implementation

## Appendix: Code Samples

### API Gateway Endpoint (Python/FastAPI)
```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis
import os

app = FastAPI()
security = HTTPBearer()
redis_client = redis.Redis(host=os.getenv('REDIS_HOST'), port=6379)

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != os.getenv('API_KEY'):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return credentials.credentials

@app.get("/redis/get/{key}")
async def redis_get(key: str, api_key: str = Depends(verify_api_key)):
    try:
        value = redis_client.get(key)
        return {"key": key, "value": value.decode() if value else None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Cerebrium Redis Client Wrapper
```python
class HTTPRedisClient:
    """Redis client that uses HTTP API instead of direct connection"""
    
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    async def get(self, key: str):
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.api_url}/redis/get/{key}",
                headers=self.headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("value")
                return None
```

---

*Document Version: 1.0*  
*Last Updated: 2024-12-07*  
*Author: Liddy Voice Team*
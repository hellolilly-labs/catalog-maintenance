# Liddy AI Deployments

This directory contains deployment configurations and scripts for all Liddy AI services.

## Directory Structure

```
deployments/
├── liddy_voice/            # Voice service deployment
│   ├── Dockerfile          # Container definition
│   ├── gcp/               # GCP-specific scripts
│   └── README.md          # Voice service deployment guide
└── README.md              # This file
```

## Services

### liddy_voice
The voice assistant service powered by LiveKit. Provides real-time voice interactions with brand-aware AI agents.

See [liddy_voice/README.md](liddy_voice/README.md) for deployment instructions.

## Common Patterns

All deployment configurations follow these patterns:

1. **Dockerfiles**: Located at `deployments/<service>/Dockerfile`
2. **Cloud Provider Scripts**: Located at `deployments/<service>/<provider>/`
3. **Build Context**: All Docker builds use the monorepo root as context
4. **Package Installation**: Dockerfiles install packages in dependency order

## Future Services

As we add more services, they will follow the same structure:

- `liddy_intelligence/` - Brand research and catalog services
- `liddy_api/` - REST API services
- `liddy_dashboard/` - Web dashboard

## Best Practices

1. **Secrets**: Never hardcode secrets in Dockerfiles or scripts
2. **Caching**: Use Docker build caching for faster deployments
3. **Security**: Run containers as non-root users
4. **Monitoring**: Include health checks and logging configuration
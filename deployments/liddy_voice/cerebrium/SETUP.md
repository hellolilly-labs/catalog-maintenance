# Quick Setup Guide for Cerebrium Deployment

## 1. Install Cerebrium CLI

```bash
pip install cerebrium
```

## 2. Login to Cerebrium

```bash
cerebrium login
```

This will open a browser for authentication.

## 3. Create Environment File

Copy your existing `.env` file or create a new one:

```bash
cd deployments/liddy_voice/cerebrium
cp ../../../.env .env
```

Make sure all required API keys are set.

## 4. Deploy to Cerebrium

Run the deployment script:

```bash
./deploy.sh
```

This will:
- Check for Cerebrium CLI
- Initialize the project
- Package the code
- Deploy to Cerebrium

## 5. Get Your Project ID

After deployment, you'll see output like:
```
Your project has been deployed at:
https://api.cortex.cerebrium.ai/v4/p-abc123/liddy-voice-agent/run
```

The `p-abc123` part is your project ID.

## 6. Test the Deployment

```bash
# Set your project ID
export CEREBRIUM_PROJECT_ID=abc123

# Run tests
python test_deployment.py
```

## 7. Monitor Logs

```bash
cerebrium logs liddy-voice-agent --tail
```

## Common Issues & Solutions

### Issue: "Module not found" errors
**Solution**: Make sure all dependencies are in `requirements-cerebrium.txt`

### Issue: Environment variables not set
**Solution**: Check that `.env` file exists and is properly formatted

### Issue: GPU out of memory
**Solution**: Reduce model sizes or batch processing

### Issue: Cold start too slow
**Solution**: Set `min_replicas: 1` in cerebrium.toml to keep one instance warm

## Next Steps

1. Configure your LiveKit server to use the Cerebrium endpoints
2. Set up monitoring and alerts
3. Test with real voice conversations
4. Optimize for your specific use case

## Support

- Join Cerebrium Discord: https://discord.gg/ATj6USmeE2
- Check Cerebrium status: https://status.cerebrium.ai
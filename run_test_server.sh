#!/bin/sh

echo "Starting test server"
uvicorn server:app --reload --host 0.0.0.0 --port 8000

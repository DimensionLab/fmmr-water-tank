FROM nvcr.io/nvidia/modulus/modulus:22.09

RUN pip install pydantic fastapi "uvicorn[standard]"
WORKDIR /app/
COPY . /app/

CMD ["./run_test_server.sh"] 
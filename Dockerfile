FROM nvcr.io/nvidia/modulus/modulus:22.09

RUN rm -rf \
    /usr/lib/x86_64-linux-gnu/libcuda.so* \
    /usr/lib/x86_64-linux-gnu/libnvcuvid.so* \
    /usr/lib/x86_64-linux-gnu/libnvidia-*.so* \
    /usr/lib/firmware \
    /usr/local/cuda/compat/lib

RUN pip install pydantic fastapi "uvicorn[standard]"
WORKDIR /app/
COPY . /app/

CMD ["./run_test_server.sh"] 
import uvicorn

if __name__ == '__main__':
    uvicorn.run("api.model:app" , reload=True, host='localhost', port=3000)

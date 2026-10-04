from api.inference_apis import app

def main():
    import uvicorn
    uvicorn.run("api.inference_apis:app",host="0.0.0.0",port=8000, reload=True)


if __name__ == "__main__":
    main()

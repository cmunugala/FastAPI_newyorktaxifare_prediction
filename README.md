# FastAPI_newyorktaxifare_prediction

In this project, I created an API for my new york taxi fare prediction model, which I created using PyTorch (see https://github.com/cmunugala/new-york-taxi-fare). This code creates a predction endpoint where the user can submit a tab seperated text file with the ten features (specified below) and the API will return the prediction for the new york taxi fare. 

Tab Seperated File Instructions:

The values for the features should be in the following order. 10 values per row. Each row represents one taxi ride. 

    passenger_count: int 
    diff_longitude: float 
    diff_latitude: float 
    morning: int 
    afternoon: int 
    night: int
    fall: int 
    spring: int 
    summer: int 
    winter: int 
    
Future Direction:
Working on putting this API into a Docker container and so its is portable and then deploying on AWS for practice. 

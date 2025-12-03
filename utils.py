from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
              
project = rf.workspace("tennis-wkg7w").project("tennis-all")
version = project.version(4)
dataset = version.download("yolov12")
                
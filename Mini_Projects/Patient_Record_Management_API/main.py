from fastapi import FastAPI, Path ,HTTPException, Query
from fastapi.responses import JSONResponse
import json
from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field, computed_field

app = FastAPI()

class Patient(BaseModel):
    id: Annotated[str,Field(...,description="ID of the patient", example="P001")]
    name: Annotated[str,Field(...,description="Name of the patient")]
    city: Annotated[str,Field(...,description="City of the patient")]
    age: Annotated[int,Field(...,gt=0,lt=120,description="Age of the patient")] 
    gender: Annotated[Literal['male','female','others'],Field(...,description="Gender of the patient")]
    height: Annotated[float,Field(...,gt=0,description="Height of the patient in meters")]
    weight: Annotated[float,Field(...,gt=0,description="Weight of the patient in kgs")]

    @computed_field     #@computed_field is a decorator that lets us define a read-only field in your model.
    @property  #@property is a built-in Python decorator that turns a method into a read-only attribute.Without it, you call methods like obj.method().With @property, you can access them like obj.attribute.
    def bmi(self) -> float:
        bmi=(self.weight/(self.height ** 2))
        return bmi
    
    @computed_field
    @property
    def verdict(self)->str:
        if self.bmi <18.5:
            return "Underweight"
        elif self.bmi < 30:
            return 'Normal'
        else :
            return 'Obese' 
        
class PatientUpdate(BaseModel):
    name: Annotated[Optional[str],Field(...,description="Name of the patient")]
    city: Annotated[Optional[str],Field(...,description="City of the patient")]
    age: Annotated[Optional[int],Field(...,gt=0,lt=120,description="Age of the patient")] 
    gender: Annotated[Optional[Literal['male','female','others']],Field(...,description="Gender of the patient")]
    height: Annotated[Optional[float],Field(...,gt=0,description="Height of the patient in meters")]
    weight: Annotated[Optional[float],Field(...,gt=0,description="Weight of the patient in kgs")]

def load_data():
    with open("patients.json","r") as f:
        data=json.load(f)
    return data

def save_data(data):
    with open('patients.json','w') as f:
        json.dump(data,f)

@app.get("/")
def hello():
    return {"message":"Patient Management System API"}

@app.get("/about")
def about():
    return {"message":"A fully functional API to manage your patient records"}

@app.get("/view")
def view():
    data=load_data()
    return data

@app.get("/patient/{patient_id}")
def view_patient(patient_id:str=Path(..., description="ID of the patient is required", example="P001")):
    data=load_data()
    if patient_id in data:
        return data[patient_id]
    raise HTTPException(status_code=404, detail="Patient not found")

@app.get("/sort")
def sort_patients(sort_by: str=Query(...,description="sort on the basis of weight,heigh or bmi"),order: str = Query('asc' ,description="sort in asc or descending order")):
    valid_fields=['height','weight','bmi']
    if sort_by not in valid_fields:
      raise HTTPException(status_code=400, details="Invalid field select from {vaild_field} ")
    
    if order not in ['asc','desc']:
        raise HTTPException(status_code=400, details="Invalid field select between asc and desc")
    data=load_data()
    sort_order= True if order=='desc' else False
    sorted_data=sorted(data.values(),key=lambda x: x.get(sort_by,0),reverse=sort_order)
    return sorted_data

@app.post('/create')
def create_patient(patient: Patient):
    data=load_data()
    if patient.id in data:
        raise HTTPException(status_code=400,detail="Patient already exist") 
 
    data[patient.id]=patient.model_dump(exclude=['id'])
    save_data(data)

    return JSONResponse(status_code=201,content={'message':'patient added successfully !'})

@app.put('/edit/{patient_id}')
def update_patient(patient_id:str, patient_update: PatientUpdate):
    data=load_data()
    if patient_id not in data:
        raise HTTPException(status_code=404,detail="Patient not found")
    existing_patient_data=data[patient_id]

    updated_info=patient_update.model_dump(exclude_unset=True)

    for key,value in updated_info.items():
        existing_patient_data[key]=value
    
    existing_patient_data['id']=patient_id
    patient_pydantic_obj = Patient(**existing_patient_data)

    existing_patient_data=patient_pydantic_obj.model_dump(exclude=['id'])

    data[patient_id]=existing_patient_data
    
    save_data(data)

    return JSONResponse(status_code=200,content={"msg":"Patient updated successfully !"})
    
@app.delete("/delete/{patient_id}")
def delete_patient(patient_id: str):
    data=load_data()
    if patient_id not in data:
        raise HTTPException(status_code=404,detail='Patient does not exist !')
    
    del data[patient_id]

    save_data(data)

    return JSONResponse(status_code=200,content={"msg":"Patient record deleted successfully !"})
    



    


from ultralytics import solutions
from ultralytics.solutions import Inference
import streamlit as st

inf = solutions.Inference(model="models/best.pt")  

inf.inference()
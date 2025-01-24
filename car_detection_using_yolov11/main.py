from ultralytics import solutions
import streamlit as st

inf = solutions.Inference(model="models/best.pt")  # Model is not necessary argument.

inf.inference()
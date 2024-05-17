# Aerial Object Segmentation

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Ideology](#project-ideology)
- [Models and Results](#models-and-results)
- [Deployment](#deployment)

## Introduction

This project aims to create a fully functional security surveillance system using computer vision (CV) to develop a detection model for airborne objects with high accuracy and low inference time.

## Dataset

The dataset used for this project was sourced from an AI Crowd competition and is hosted on an AWS S3 bucket. Key details include:
- **Total Size:** 11 TB
- **Content:** Black and white videos of small incoming airborne objects (planes and helicopters)
- **Selected Subset:** 120 high-quality videos, 10,000 images
- **Annotation:** Dataset was annotated for segmentation using LabelMe.

## Project Ideology

### Initial Attempts
- **YOLOv8 Detection Models:** Initial attempts with YOLOv8 detection models resulted in a low mAP (Mean Average Precision) of 55 due to incorrect classifications.

### Improved Approach
- **Segmentation Model:** Switching to a segmentation model significantly improved accuracy, achieving a mAP of 62.
- **Image Filters:** Applied various image filters to enhance the mAP, with Sobel filters yielding the best results, increasing the mAP to 65.
- **Dataset Division:** The dataset was intrinsically divided into large and small objects due to the model's initial struggle with large objects, resulting in an improved mAP of 68 and better performance on test data.

## Models and Results

1. **YOLOv8 Detection Models:**
   - **Initial mAP:** 55%
   - **Issues:** Incorrect classification.

2. **Segmentation Model:**
   - **Initial mAP:** 62%
   - **Improved mAP with Sobel Filters:** 65

3. **Dataset Division (Large and Small Objects):**
   - **Final mAP:** 68%
   - **Test Performance:** Improved significantly.

## Deployment

The final model was deployed using Flask to provide a web interface, enabling users to interact with the system and upload videos for airborne object detection.



# Plant Disease Classification using Vision Transformer (ViT)

![Built With](https://img.shields.io/badge/Built%20With-PyTorch%2C%20Timm%2C%20Torchvision-blue)  
![Language](https://img.shields.io/badge/Language-Python-orange)  
![Status](https://img.shields.io/badge/Project%20Status-Complete-brightgreen)

## Why This Project Matters

With the global population expected to exceed 9 billion by 2050, food demand is projected to rise by 70%, making early disease detection critical for food security (Kaggle, n.d).

Manual inspection of crops is time-consuming, error-prone, and unsustainable at scale. This project explores how AI—specifically, **Vision Transformers (ViT)**—can transform agriculture by detecting diseases from leaf images with **99.88% test accuracy**.

Using the **PlantVillage dataset** from Kaggle (50,000+ labeled leaf images), I trained a ViT model that segments images into patches and processes them as sequences—capturing complex spatial patterns that traditional CNNs often miss.

This solution demonstrates how AI can:
- Prevent major crop losses  
- Enable early intervention  
- Support precision farming at scale  

Looking ahead, this model could be adapted for **mobile or drone deployment** and enhanced with **real-time environmental data**.

## Dataset Preparation

- **Source**: [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Classes**: 15 categories (e.g., early blight, bacterial spot, healthy)
- **Split**: 80% training, 20% validation using `splitfolders`

## Image Preprocessing

- All images resized to **224x224**
- Transformations applied:
  - Random horizontal flips  
  - Random rotations  
  - Color jittering  
  - Normalization (ImageNet mean/std)

## Model Architecture

- **Backbone**: `vit_base_patch16_224` from `timm`
- **Fine-tuning**: Classification head updated for 15 classes
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: AdamW

## Training Setup

- **Epochs**: 10  
- **Batch size**: 32  
- **Device**: GPU (if available), else CPU  
- **Performance**: Reduced training loss from **1.3065** to **0.0012**

## Evaluation & Results

The table below shows per-class accuracy based on validation set predictions:

| Class                                             | Support | Correct Predictions | Accuracy (%) |
|--------------------------------------------------|---------|----------------------|---------------|
| Pepper__bell___Bacterial_spot                    | 200     | 200                  | 100.00        |
| Pepper__bell___healthy                           | 296     | 296                  | 100.00        |
| Potato___Early_blight                            | 200     | 200                  | 100.00        |
| Potato___Late_blight                             | 200     | 200                  | 100.00        |
| Potato___healthy                                  | 31      | 31                   | 100.00        |
| Tomato_Bacterial_spot                            | 426     | 426                  | 100.00        |
| Tomato_Early_blight                              | 198     | 197                  | 99.49         |
| Tomato_Late_blight                               | 382     | 382                  | 100.00        |
| Tomato_Leaf_Mold                                 | 191     | 191                  | 100.00        |
| Tomato_Septoria_leaf_spot                        | 355     | 355                  | 100.00        |
| Tomato_Spider_mites_Two_spotted_spider_mite      | 336     | 336                  | 100.00        |
| Tomato__Target_Spot                              | 280     | 279                  | 99.64         |
| Tomato__Tomato_YellowLeaf__Curl_Virus            | 642     | 642                  | 100.00        |
| Tomato__Tomato_mosaic_virus                      | 75      | 75                   | 100.00        |
| Tomato_healthy                                   | 319     | 319                  | 100.00        |

## Conclusion 

This project demonstrates the powerful potential of Vision Transformers in agricultural diagnostics. By achieving 99.88% accuracy across 15 classes of plant health conditions, the model proves highly reliable in identifying diseases from leaf imagery.

Beyond academic success, this approach offers real-world value by helping farmers and agricultural experts:
- Detect diseases early  
- Reduce crop loss  
- Improve productivity and sustainability  

## Next Steps

To further enhance the model's practicality and impact, the following improvements are recommended:

- **Expand Dataset**: Include more crop types and real-world farm conditions to improve robustness.
- **Integrate Environmental Data**: Combine image analysis with factors like soil quality, temperature, and humidity for more holistic predictions.
- **Deploy on Mobile & Drones**: Optimize for low-resource environments to enable real-time, in-field disease detection using smartphones or agricultural drones.
- **Multilingual User Interface**: Make the tool accessible to farmers globally by supporting local languages.
- **Model Explainability**: Add SHAP or Grad-CAM visualizations to help farmers and researchers understand why the model makes specific predictions.

These enhancements can transform this AI solution into a powerful tool for modern, sustainable agriculture.

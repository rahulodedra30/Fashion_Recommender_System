# Fashion Recommender System

This project implements a Fashion Recommender System using Convolutional Neural Networks (CNN) and Transfer Learning with ResNET architecture. The system enables reverse image search and groups visually similar products based on extracted image features.

### Key Features:
- Feature Extraction: Fine-tuned a ResNET model to extract high-dimensional features from a dataset of 44,000 fashion images.
- Recommendation Generation: Utilized K-Nearest Neighbors (KNN) algorithm to generate recommendations based on Euclidean distance between feature vectors.
- Streamlit Interface: Integrated with a Streamlit interface for user-friendly interaction and visual display of recommendations.

### Run Streamlit app:
```bash
   streamlit run main.py
```

### Recommendations sample:
The image uploaded by the user is displayed along with the 5 nearest recommendations below it.


![Blue Tshirt recommendation](https://github.com/rahulodedra30/Fashion_Recommender_System/blob/main/results/result1.png)
![](https://github.com/rahulodedra30/Fashion_Recommender_System/blob/main/results/result2.png)

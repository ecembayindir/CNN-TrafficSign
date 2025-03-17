<h1 align="center">Traffic Sign Classification with CNNs</h1>
<br/>

<h2>🚀 <strong>Objective</strong></h2>
<p>
    The goal of this project is to develop a robust traffic sign classification system using Convolutional Neural Networks (CNNs). 
    The system leverages multiple deep learning approaches, including custom CNN architectures, data augmentation, hyperparameter tuning, 
    and transfer learning with VGG19, to accurately classify 43 distinct traffic sign classes from the German Traffic Sign Recognition 
    Benchmark (GTSRB) dataset.
</p>

<h2>🛠 Technology Stack</h2>
<div class="badges">
    <img src="https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python"/>
    <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
    <img src="https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras"/>
      <img src="https://img.shields.io/badge/Scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn"/>
    <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
    <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
    <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib"/>
    <img src="https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=seaborn&logoColor=white" alt="Seaborn"/>
    <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/>
    <img src="https://img.shields.io/badge/Keras_Tuner-FF6F61?style=for-the-badge" alt="Keras Tuner"/>
</div>

<h2>📂 <strong>Project Summary</strong></h2>
<p>
    This project implements a traffic sign classification system using the GTSRB dataset. It explores various CNN-based approaches, 
    including a baseline CNN, an enhanced CNN with data augmentation and regularization, a hyperparameter-tuned model, and transfer 
    learning with VGG19. The best-performing model achieves a validation accuracy of 99.69%, demonstrating high precision, recall, 
    and F1-scores across 43 classes, making it suitable for real-world traffic sign recognition tasks.
</p>

<h2>📈 <strong>Methodology</strong></h2>
<table style="width:100%; border-collapse: collapse; margin-bottom: 20px;">
    <thead>
        <tr>
            <th style="background-color: #f2f2f2; border: 1px solid #ddd; padding: 8px;">Approach</th>
            <th style="background-color: #f2f2f2; border: 1px solid #ddd; padding: 8px;">Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">Base CNN Model</td>
            <td style="border: 1px solid #ddd; padding: 8px;">Three convolutional blocks with increasing filters (32, 64, 128) to capture spatial hierarchies.</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">Data Augmentation</td>
            <td style="border: 1px solid #ddd; padding: 8px;">Applied rotation, shifts, zoom, and flips to enhance model robustness to variations.</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">Enhanced Model</td>
            <td style="border: 1px solid #ddd; padding: 8px;">Added batch normalization and strategic dropout layers for improved regularization.</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">Hyperparameter Tuning</td>
            <td style="border: 1px solid #ddd; padding: 8px;">Used Keras Tuner to systematically search for optimal model parameters.</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">Transfer Learning</td>
            <td style="border: 1px solid #ddd; padding: 8px;">Leveraged VGG19 pre-trained on ImageNet with custom classification layers.</td>
        </tr>
    </tbody>
</table>

<h2>🔍 <strong>Implementation Details</strong></h2>

<h3>📋 Data Preprocessing</h3>
<table style="width:100%; border-collapse: collapse; margin-bottom: 20px;">
    <thead>
        <tr>
            <th style="background-color: #f2f2f2; border: 1px solid #ddd; padding: 8px;">Step</th>
            <th style="background-color: #f2f2f2; border: 1px solid #ddd; padding: 8px;">Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">1</td>
            <td style="border: 1px solid #ddd; padding: 8px;">Loaded GTSRB dataset metadata from Train.csv and Test.csv.</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">2</td>
            <td style="border: 1px solid #ddd; padding: 8px;">Resized images to 32x32 pixels and normalized pixel values to [0, 1].</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">3</td>
            <td style="border: 1px solid #ddd; padding: 8px;">Converted images to RGB and one-hot encoded labels for 43 classes.</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">4</td>
            <td style="border: 1px solid #ddd; padding: 8px;">Split data into training (31,367 samples) and validation (7,842 samples) sets.</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">5</td>
            <td style="border: 1px solid #ddd; padding: 8px;">Applied data augmentation (rotation, shift, zoom, flip) to training set.</td>
        </tr>
    </tbody>
</table>

<h3>📌 Model Development</h3>
<ul>
    <li>Baseline CNN: 3 Conv2D layers (32, 64, 128 filters), MaxPooling, Dense layers (256, 43 units).</li>
    <li>Enhanced CNN: Added BatchNormalization and Dropout (0.2-0.4) with data augmentation.</li>
    <li>Tuned CNN: Optimized with Keras Tuner (filters: 64, dropout: 0.2, units: 512, learning rate: 0.0001).</li>
    <li>VGG19: Pre-trained model with custom dense layers (256, 43 units) and fine-tuning of last 4 layers.</li>
</ul>

<h3>📌 Evaluation Metrics</h3>
<table style="width:100%; border-collapse: collapse; margin-bottom: 20px;">
    <thead>
        <tr>
            <th style="background-color: #f2f2f2; border: 1px solid #ddd; padding: 8px;">Model</th>
            <th style="background-color: #f2f2f2; border: 1px solid #ddd; padding: 8px;">Accuracy</th>
            <th style="background-color: #f2f2f2; border: 1px solid #ddd; padding: 8px;">Loss</th>
            <th style="background-color: #f2f2f2; border: 1px solid #ddd; padding: 8px;">Precision</th>
            <th style="background-color: #f2f2f2; border: 1px solid #ddd; padding: 8px;">Recall</th>
            <th style="background-color: #f2f2f2; border: 1px solid #ddd; padding: 8px;">F1-Score</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">Base CNN</td>
            <td style="border: 1px solid #ddd; padding: 8px;">98.19%</td>
            <td style="border: 1px solid #ddd; padding: 8px;">0.0731</td>
            <td style="border: 1px solid #ddd; padding: 8px;">0.9857</td>
            <td style="border: 1px solid #ddd; padding: 8px;">0.9834</td>
            <td style="border: 1px solid #ddd; padding: 8px;">0.9837</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">CNN with Augmentation</td>
            <td style="border: 1px solid #ddd; padding: 8px;">95.55%</td>
            <td style="border: 1px solid #ddd; padding: 8px;">0.1290</td>
            <td style="border: 1px solid #ddd; padding: 8px;">0.9535</td>
            <td style="border: 1px solid #ddd; padding: 8px;">0.9514</td>
            <td style="border: 1px solid #ddd; padding: 8px;">0.9500</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">Best Tuned Model</td>
            <td style="border: 1px solid #ddd; padding: 8px;">99.69%</td>
            <td style="border: 1px solid #ddd; padding: 8px;">0.0131</td>
            <td style="border: 1px solid #ddd; padding: 8px;">0.9970</td>
            <td style="border: 1px solid #ddd; padding: 8px;">0.9972</td>
            <td style="border: 1px solid #ddd; padding: 8px;">0.9970</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">VGG19 Transfer Learning</td>
            <td style="border: 1px solid #ddd; padding: 8px;">64.96%</td>
            <td style="border: 1px solid #ddd; padding: 8px;">1.0622</td>
            <td style="border: 1px solid #ddd; padding: 8px;">0.6597</td>
            <td style="border: 1px solid #ddd; padding: 8px;">0.5614</td>
            <td style="border: 1px solid #ddd; padding: 8px;">0.5788</td>
        </tr>
    </tbody>
</table>

<h2>📊 <strong>Key Improvements</strong></h2>
<ul>
    <li><strong>Accuracy:</strong> Achieved 99.69% validation accuracy with the hyperparameter-tuned model.</li>
    <li><strong>Robustness:</strong> Data augmentation and regularization reduced overfitting and improved generalization.</li>
    <li><strong>Efficiency:</strong> Optimized model parameters using Keras Tuner for better performance with fewer epochs.</li>
    <li><strong>Interpretability:</strong> Visualized layer activations and misclassified samples for model analysis.</li>
</ul>

<h2>🔍 <strong>Implementation Workflow</strong></h2>
<ol>
    <li>Loaded and preprocessed GTSRB dataset images and metadata.</li>
    <li>Developed and trained a baseline CNN model.</li>
    <li>Enhanced the model with data augmentation, batch normalization, and dropout.</li>
    <li>Tuned hyperparameters using Keras Tuner to optimize performance.</li>
    <li>Applied transfer learning with VGG19 and fine-tuned the model.</li>
    <li>Evaluated all models using accuracy, precision, recall, F1-score, and loss metrics.</li>
</ol>

<h2>📚 <strong>Dataset</strong></h2>
<p>
    The project uses the <strong>German Traffic Sign Recognition Benchmark (GTSRB)</strong> dataset, a widely-used benchmark 
    for traffic sign classification. It contains over 50,000 images of 43 traffic sign classes, collected under various 
    lighting and weather conditions.
</p>
<ul>
    <li><strong>Source:</strong> <a href="https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign">GTSRB on Kaggle</a></li>
    <li><strong>Contents:</strong> 39,209 training images and 12,630 test images with metadata (e.g., width, height, ROI, class ID).</li>
    <li><strong>Size:</strong> Over 50,000 images resized to 32x32 pixels.</li>
    <li><strong>License:</strong> Free for academic use.</li>
</ul>

<h2>📢 <strong>Conclusion</strong></h2>
<p>
    This project successfully developed a high-performing traffic sign classification system, with the hyperparameter-tuned 
    CNN achieving a 99.69% accuracy. Key findings include the effectiveness of data augmentation and regularization, the 
    superior performance of custom-tuned CNNs over transfer learning with VGG19 for this dataset, and the importance of 
    hyperparameter optimization. The system is scalable and adaptable for real-time traffic sign recognition applications.
</p>

<h2>Connect With Me</h2>
<div align="center">
    <a href="https://www.linkedin.com/in/ecembayindir" target="_blank">
        <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/>
    </a>
    <a href="mailto:ecmbyndr@gmail.com">
        <img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email"/>
    </a>
</div>
<br>
<p align="center">© 2025 Ecem Bayindir. All rights reserved.</p>
<hr/>
<p align="center">
    <img src="https://komarev.com/ghpvc/?username=ecembayindir&repo=CNN-TrafficSign&label=Repository%20views&color=0e75b6&style=flat" alt="Repository Views">
</p>

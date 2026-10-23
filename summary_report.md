# Summary Report – Deep Learning Practical Exam (Set A)

**Name:** Krisha Anghan

## The Five Tasks and Datasets

I built five standalone mini-projects. The first was an **ANN** that predicts diabetes on the Pima Indians Diabetes dataset (768 patients, 8 features). The second was an **OpenCV** preprocessing pipeline run on five images from the Flowers Recognition dataset. The third was a **CNN** that classifies CIFAR-10 images into 10 classes, using 20,000 training and 4,000 test samples. The fourth was a **SimpleRNN** that forecasts next month's airline passengers from the previous 12 months (144 months of data). The fifth used a pre-trained **DistilBERT Transformer** from Hugging Face to classify the sentiment of 25 movie reviews that I wrote myself.

## Results

- **ANN:** 70.78% test accuracy, with precision 0.58, recall 0.63 and F1 0.60. The confusion matrix showed 75 true negatives, 34 true positives, 25 false positives and 20 false negatives. This is within the 70–80% target.
- **OpenCV:** All five images went through resizing, grayscale, Gaussian blur, Canny edges and largest-contour detection, and the results were shown in a 5×5 grid.
- **CNN:** 64.40% test accuracy, inside the 55–70% target. Automobile, ship and frog were classified best, while cat, bird and dog were the hardest classes.
- **RNN:** Test RMSE of 40.23 passengers on a series that ranges from about 100 to 600.
- **Transformer:** 19 of 20 clear reviews were correct (95%). The only mistake was "I would never recommend this movie", where the model ignored the negation.

## What Was New to Me

The most interesting concept was the **Transformer pipeline**. I learned that a pre-trained, fine-tuned model can be used directly without any training. I also learned its weaknesses. The model is binary, so it must pick positive or negative even for neutral sentences, and its confidence score stays very high (often above 0.98) on ambiguous text. On Indian-English text it gave a confident but wrong answer for "Paisa vasool movie", which means "worth the money" and is actually positive. I also learned in the RNN task why time-series data must never be shuffled, and how OpenCV steps such as blur and edge detection relate to the low-level features a CNN learns on its own.

## Challenges

The CNN was the hardest step because training took the longest and the architecture had to be chosen carefully. For the ANN, replacing the physiologically impossible zero values with column medians was important before training.

## What I Would Build Next

With more time and compute I would:

1. Improve the **ANN** with Dropout, early stopping and class weighting, because recall for diabetic patients matters most in a medical setting.
2. Upgrade the **CNN** with data augmentation, batch normalisation, more epochs and the full 50,000 images, or use transfer learning with a model such as MobileNet. I would then deploy it with FastAPI or Streamlit.
3. Replace the **SimpleRNN** with an **LSTM** and compare the RMSE.
4. Fine-tune a **multilingual Transformer** (for example XLM-R or MuRIL) on Hindi-English and Gujarati-English reviews so that slang and code-mixed text is handled properly.

Overall, the exam gave me hands-on experience with the full workflow: loading data, building, training, evaluating and saving models for tabular, image, sequence and text data.

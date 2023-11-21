import pandas as pd 
import os 
from skimage.transform import resize 
from skimage.io import imread 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import svm 
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
import seaborn as sns

Categories=['happy','sad'] 
flat_data_arr=[] #input array 
target_arr=[] #output array 
datadir='train/'
# path which contains all the categories of images
for i in Categories:
    print(f'loading... category : {i}')
    path = os.path.join(datadir, f'{i}_train')  # corrected directory path
    for img in os.listdir(path):
        img_array = imread(os.path.join(path, img))
        img_resized = resize(img_array, (48, 48, 3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(i))
    print(f'loaded category:{i} successfully')

flat_data = np.array(flat_data_arr)
target = np.array(target_arr)

#dataframe 
df=pd.DataFrame(flat_data) 
df['Target']=target 
df.shape

#input data 
x=df.iloc[:,:-1] 
#output data 
y=df.iloc[:,-1]

# Splitting the data into training and testing sets 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20, 
											random_state=123, 
											stratify=y) 

# Defining the parameters grid for GridSearchCV 
param_grid={'C':[0.1,1,10,100], 
			'gamma':[0.0001,0.001,0.1,1], 
			'kernel':['rbf','poly']} 

# Creating a support vector classifier 
svc=svm.SVC(probability=True) 

# Creating a model using GridSearchCV with the parameters grid 
model=GridSearchCV(svc,param_grid)

# Training the model and storing the learning curve
train_sizes, train_scores, validation_scores = learning_curve(model, x_train, y_train, cv=5, scoring='accuracy')

# Plotting the learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Accuracy')
plt.plot(train_sizes, np.mean(validation_scores, axis=1), label='Validation Accuracy')
plt.title('Learning Curve')
plt.xlabel('Training Examples')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Training the model using the training data
model.fit(x_train, y_train)

def predict_on_test_dataset(model, Categories, test_dir='test/'):
    true_labels = []
    predicted_labels = []

    for category in Categories:
        category_path = os.path.join(test_dir, f'{category}_test')
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            
            # Read the image
            img = imread(img_path)
            
            # Display the image
            plt.imshow(img)
            plt.title(f'Category: {category}')
            plt.show()
            
            # Resize the image
            img_resize = resize(img, (48, 48, 3))
            
            # Flatten the image data
            img_flat = img_resize.flatten()
            
            # Make a prediction
            prediction = model.predict([img_flat])[0]
            probability = model.predict_proba([img_flat])
            
            # Display the prediction probabilities
            for ind, val in enumerate(Categories):
                print(f'{val} = {probability[0][ind] * 100}%')
            
            # Append true and predicted labels
            true_labels.append(Categories.index(category))
            predicted_labels.append(prediction)

            # Display the predicted image category
            print("The predicted image is: " + Categories[prediction])

    from sklearn.metrics import confusion_matrix, accuracy_score    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Display confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=Categories, yticklabels=Categories)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Display classification report
    report = classification_report(true_labels, predicted_labels, target_names=Categories)
    print('Classification Report:\n', report)

    print(f'Accuracy: {accuracy * 100:.2f}%')

# Call the function with your model and Categories
predict_on_test_dataset(model, Categories)

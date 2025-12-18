The following code demonstrates various neural network architectures used on the `medical students dataset` for diabetes prediction.


```python
# Base FNN (2-hidden layers)
class FNN_2Layer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# FNN with 3 hidden layers
class FNN_3Layer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# FNN with dropout (regularized)
class FNN_Dropout(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# Wider but shallow FNN
class FNN_Wide(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

```


after the training this result was found:
```csv
scaler,model,val_accuracy,features
standard,fnn_dropout,0.8996643771464252,"['Temperature', 'Height', 'Gender_Male', 'Age', 'Cholesterol', 'Blood Type_AB', 'Heart Rate', 'Smoking_Yes', 'Blood Pressure', 'Blood Type_O']"
minmax,fnn_dropout,0.8996643771464252,"['Temperature', 'Height', 'Gender_Male', 'Age', 'Cholesterol', 'Blood Type_AB', 'Heart Rate', 'Smoking_Yes', 'Blood Pressure', 'Blood Type_O']"
minmax,fnn_wide,0.896034967218233,"['Temperature', 'Height', 'Gender_Male', 'Age', 'Cholesterol', 'Blood Type_AB', 'Heart Rate', 'Smoking_Yes', 'Blood Pressure', 'Blood Type_O']"
minmax,fnn_2layer,0.8904152357165158,"['Temperature', 'Height', 'Gender_Male', 'Age', 'Cholesterol', 'Blood Type_AB', 'Heart Rate', 'Smoking_Yes', 'Blood Pressure', 'Blood Type_O']"
minmax,fnn_3layer,0.8795660318451451,"['Temperature', 'Height', 'Gender_Male', 'Age', 'Cholesterol', 'Blood Type_AB', 'Heart Rate', 'Smoking_Yes', 'Blood Pressure', 'Blood Type_O']"
standard,fnn_wide,0.8438182953481111,"['Temperature', 'Height', 'Gender_Male', 'Age', 'Cholesterol', 'Blood Type_AB', 'Heart Rate', 'Smoking_Yes', 'Blood Pressure', 'Blood Type_O']"
standard,fnn_3layer,0.8020995941305027,"['Temperature', 'Height', 'Gender_Male', 'Age', 'Cholesterol', 'Blood Type_AB', 'Heart Rate', 'Smoking_Yes', 'Blood Pressure', 'Blood Type_O']"
standard,fnn_2layer,0.7688495160786762,"['Temperature', 'Height', 'Gender_Male', 'Age', 'Cholesterol', 'Blood Type_AB', 'Heart Rate', 'Smoking_Yes', 'Blood Pressure', 'Blood Type_O']"

```

and then the best model was selected based on validation set accuracy. It was run on the test set and this was the result:

```shell
Final Test Set Performance:
Accuracy  : 0.8997
Precision : 0.0000
F1-score  : 0.0000
AUROC     : 0.5030
```

the roc curve is available at: /final-model-ROC.png

the training and validation loss plot for each of the model is available at:
for minmax scaler:
/minmax/{model_name.png}
for standard scaler:
standard/{model_name.png}



Your task is to create a short report in `latex` on the neural network architectures I have experimented on. For each architecture,
mention the number of layers along with the number of neurons in each layer contains, loss functions, activation functions, etc. Also, include the training and validation loss plot for each model (for the top 4 performing model only), and justify your choice of the final selected architecture.

About the title page:

<Give an appopriate title>
Name: Zia Ul Hassan Abdullah
Id: 2005037
Bangladesh University of Engineering and Technology


from flask import Flask,render_template,request
from data_cleaning import clean_words
import joblib




class Check():
    def __init__(self):
        self.lr_pipe = joblib.load('logistic_regression_fnd.pkl')
        self.dtc_pipe = joblib.load('decision_tree_classifier_fnd.pkl')
        self.gdc_pipe = joblib.load('gradient_boosting_classifier_fnd.pkl')
        self.rfc_pipe = joblib.load('random_forest_classifier_fnd.pkl')

    def lr(self,cleaned_news):
        result = self.lr_pipe.predict([cleaned_news])
        # print("clean_ip = ",cleaned_news)
        # print(result)
        return result[0]

    def dtc(self,cleaned_news):
        # cleaned_news = clean_words(news)
        result = self.dtc_pipe.predict([cleaned_news])
        return result[0]

    def gbc(self,cleaned_news):
        # cleaned_news = clean_words(news)
        result = self.gdc_pipe.predict([cleaned_news])
        return result[0]

    def rfc(self,cleaned_news):
        # cleaned_news = clean_words(news)
        result = self.rfc_pipe.predict([cleaned_news])
        return result[0]

C = Check()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html', title = 'Home')

@app.route('/home',)
def home():
    return render_template('home.html',title ='Home')

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        model = request.form['model']
        news = request.form['news']
        cleaned_news = clean_words(news)
        # print(model)
        if(not (cleaned_news  and not cleaned_news .isspace())):
            return render_template('home.html', title = 'Home',enteredmodel = model, prediction=None)

        elif cleaned_news:
            if model == "Logistic Regression":
                # print("in lr model")
                ans = C.lr(news)
                # print(ans)
                return render_template('home.html', title = 'Home',prediction=ans,
                                   enteredmodel = model, enterednews = news)
            elif model == "Decision Tree Classifier":
                # print("in dtc model")
                ans = C.dtc(news)
                # print(ans)
                return render_template('home.html', title = 'Home',prediction=ans,
                                   enteredmodel = model, enterednews = news)
            elif model == "Gradient Boosting Classifier":
                # print("in gbc model")
                ans = C.gbc(news)
                # print(ans)
                return render_template('home.html', title = 'Home',prediction=ans,
                                   enteredmodel = model, enterednews = news)
            elif model == "Random Forest Classifier":
                # print("in rfc model")
                ans = C.rfc(news)
                # print(ans)
                return render_template('home.html', title = 'Home',prediction=ans,
                                   enteredmodel = model, enterednews = news)
        else:
            return render_template('home.html',prediction = None)

@app.route('/how-it-works')
def workflow():
    return render_template('how-it-works.html',title = 'How It Works?')

@app.route('/project-report')
def project_report():
    return render_template('project-report.html',title = 'Project Report')

@app.route('/about')
def about():
    return render_template('about.html',title = 'About')

if __name__ == '__main__':
    app.run(debug=True)
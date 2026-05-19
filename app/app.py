from flask import Flask , render_template

#creating a flask app
app = Flask(__name__)


@app.route('/') #means attach this function to the url route
def home():
    return render_template("landingPage.html")


@app.route('/heart')
def heart():
    return render_template("heart.html")

@app.route('/cancer')
def cancer():
    return render_template("cancer.html")


@app.route('/diabetes')
def diabetes():
    return render_template("diabetes.html")



if __name__ == "__main__":
    app.run(debug=True)
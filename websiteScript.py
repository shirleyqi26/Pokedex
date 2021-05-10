from flask import Flask, redirect, url_for, render_template, request, Markup
from flask_ngrok import run_with_ngrok
import requests, re, time
import torch, torchvision
from torch import nn, optim
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from google.colab import files
from PIL import Image
import numpy
import cv2

device = torch.device('cuda:0') 
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 75)
model.to(device)
model.load_state_dict(torch.load('websitestuff/model'))
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

pokemonDict = { "1":"Bulbasaur", "2":"Ivysaur", "3":"Venusaur", "4":"Charmander", "5":"Charmeleon", "6":"Charizard", "7":"Squirtle", "8":"Wartortle", "9":"Blastoise", "10":"Caterpie", "11":"Metapod", "12":"Butterfree", "13":"Weedle", "14":"Kakuna", "15":"Beedrill", "16":"Pidgey", "17":"Pidgeotto", "18":"Pidgeot", "19":"Rattata", "20":"Raticate", "21":"Spearow", "22":"Fearow", "23":"Ekans", "24":"Arbok", "25":"Pikachu", "26":"Raichu", "27":"Sandshrew", "28":"Sandslash", "29":"Nidoran ♀", "30":"Nidorina", "31":"Nidoqueen", "32":"Nidoran ♂", "33":"Nidorino", "34":"Nidoking", "35":"Clefairy", "36":"Clefable", "37":"Vulpix", "38":"Ninetales","39":"Jigglypuff","40":"Wigglytuff","41":"Zubat","42":"Golbat","43":"Oddish","44":"Gloom","45":"Vileplume","46":"Paras","47":"Parasect","48":"Venonat","49":"Venomoth","50":"Diglett","51":"Dugtrio","52":"Meowth", "53":"Persian", "54":"Psyduck","55":"Golduck", "56":"Mankey","57":"Primeape","58":"Growlithe","59":"Arcanine","60":"Poliwag","61":"Poliwhirl","62":"Poliwrath","63":"Abra","64":"Kadabra","65":"Alakazam","66":"Machop","67":"Machoke","68":"Machamp","69":"Bellsprout","70":"Weepinbell","71":"Victreebel","72":"Tentacool","73":"Tentacruel","74":"Geodude","75":"Graveler"}

ALLOWED_FILES = {'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_FILES

def makePrediction(image):
  xform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), torchvision.transforms.RandomHorizontalFlip(p=0.5)])
  image = xform(image).to(device)
  model.eval()
  image = image.unsqueeze(0)
  output = model(image)
  _, pred = torch.max(output.detach(), 1)
  percent, preds = torch.topk(output.detach(), 5)
  return pred.item()+1, torch.add(preds, 1), percent

app = Flask(__name__)

run_with_ngrok(app)

@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "POST":
      if 'im' not in request.files:
            return render_template("Home.html")
      img = request.files['im']
      if img.filename == '':
            return render_template("Home.html")
      if img and allowed_file(img.filename):
            pred, Preds, Percents = makePrediction(Image.open(img))
            if pred == 1:
                return redirect(url_for("bulbasaur", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 2:
                return redirect(url_for("ivysaur", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 3:
                return redirect(url_for("venusaur", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 4:
                return redirect(url_for("charmander", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 5:
                return redirect(url_for("charmeleon", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 6:
                return redirect(url_for("charizard", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 7:
                return redirect(url_for("squirtle", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 8:
                return redirect(url_for("wartortle", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 9:
                return redirect(url_for("blastoise", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 10:
                return redirect(url_for("caterpie", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 11:
                return redirect(url_for("metapod", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 12:
                return redirect(url_for("butterfree", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 13:
                return redirect(url_for("weedle", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 14:
                return redirect(url_for("kakuna", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 15:
                return redirect(url_for("beedrill", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 16:
                return redirect(url_for("pidgey", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 17:
                return redirect(url_for("pidgeotto", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 18:
                return redirect(url_for("pidgeot", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 19:
                return redirect(url_for("rattata", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 20:
                return redirect(url_for("raticate", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 21:
                return redirect(url_for("spearow", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 22:
                return redirect(url_for("fearow", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 23:
                return redirect(url_for("ekans", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 24:
                return redirect(url_for("arbok", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 25:
                return redirect(url_for("pikachu", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 26:
                return redirect(url_for("raichu", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 27:
                return redirect(url_for("sandshrew", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 28:
                return redirect(url_for("sandslash", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 29:
                return redirect(url_for("nidoranF", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 30:
                return redirect(url_for("nidorina", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 31:
                return redirect(url_for("nidoqueen", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 32:
                return redirect(url_for("nidoranM", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 33:
                return redirect(url_for("nidorino", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 34:
                return redirect(url_for("nidoking", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 35:
                return redirect(url_for("clefairy", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 36:
                return redirect(url_for("clefable", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 37:
                return redirect(url_for("vulpix", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 38:
                return redirect(url_for("ninetales", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 39:
                return redirect(url_for("jigglypuff", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 40:
                return redirect(url_for("wigglytuff", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 41:
                return redirect(url_for("zubat", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 42:
                return redirect(url_for("golbat", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 43:
                return redirect(url_for("oddish", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 44:
                return redirect(url_for("gloom", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 45:
                return redirect(url_for("vileplume", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 46:
                return redirect(url_for("paras", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 47:
                return redirect(url_for("parasect", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 48:
                return redirect(url_for("venonat", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))                                                                              
            elif pred == 49:
                return redirect(url_for("venomoth", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 50:
                return redirect(url_for("diglett", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 51:
                return redirect(url_for("dugtrio", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 52:
                return redirect(url_for("meowth", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 53:
                return redirect(url_for("persian", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 54:
                return redirect(url_for("psyduck", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 55:
                return redirect(url_for("golduck", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 56:
                return redirect(url_for("mankey", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 57:
                return redirect(url_for("primeape", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 58:
                return redirect(url_for("growlithe", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 59:
                return redirect(url_for("arcanine", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 60:
                return redirect(url_for("poliwag", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 61:
                return redirect(url_for("poliwhirl", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 62:
                return redirect(url_for("poliwrath", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 63:
                return redirect(url_for("abra", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 64:
                return redirect(url_for("kadabra", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 65:
                return redirect(url_for("alakazam", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 66:
                return redirect(url_for("machop", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 67:
                return redirect(url_for("machoke", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 68:
                return redirect(url_for("machamp", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 69:
                return redirect(url_for("bellsprout", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 70:
                return redirect(url_for("weepinbell", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 71:
                return redirect(url_for("victreebel", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 72:
                return redirect(url_for("tentacool", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 73:
                return redirect(url_for("tentacruel", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 74:
                return redirect(url_for("geodude", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            elif pred == 75:
                return redirect(url_for("graveler", percent1 = Percents[0][0].item(), percent2 = Percents[0][1].item(), percent3 = Percents[0][2].item(), percent4 = Percents[0][3].item(), percent5 = Percents[0][4].item(), number1 = Preds[0][0].item(), number2 = Preds[0][1].item(), number3 = Preds[0][2].item(), number4 = Preds[0][3].item(), number5 = Preds[0][4].item()))
            else:
                return render_template("Home.html")
      else:
            return render_template("Home.html")
    else:
      return render_template("Home.html")
    
@app.route("/bulbasaur/")
def bulbasaur():
    return render_template("01-Bulbasaur.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/ivysaur/")
def ivysaur():
    return render_template("02-Ivysaur.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/venusaur/")
def venusaur():
    return render_template("03-Venusaur.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/charmander/")
def charmander():
    return render_template("04-Charmander.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/charmeleon/")
def charmeleon():
    return render_template("05-Charmeleon.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/charizard/")
def charizard():
    return render_template("06-Charizard.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/squirtle/")
def squirtle():
    return render_template("07-Squirtle.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/wartortle/")
def wartortle():
    return render_template("08-Wartortle.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/blastoise/")
def blastoise():
    return render_template("09-Blastoise.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])  

@app.route("/caterpie/")
def caterpie():
    return render_template("10-Caterpie.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/metapod/")
def metapod():
    return render_template("11-Metapod.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/butterfree/")
def butterfree():
    return render_template("12-Butterfree.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/weedle/")
def weedle():
    return render_template("13-Weedle.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/kakuna/")
def kakuna():
    return render_template("14-Kakuna.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/beedrill/")
def beedrill():
    return render_template("15-Beedrill.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/pidgey/")
def pidgey():
    return render_template("16-Pidgey.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/pidgeotto/")
def pidgeotto():
    return render_template("17-Pidgeotto.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/pidgeot/")
def pidgeot():
    return render_template("18-Pidgeot.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/rattata/")
def rattata():
    return render_template("rattatatest.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/raticate/")
def raticate():
    return render_template("20-Raticate.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])  

@app.route("/spearow/")
def spearow():
    return render_template("21-Spearow.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/fearow/")
def fearow():
    return render_template("22-Fearow.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/ekans/")
def ekans():
    return render_template("23-Ekans.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/arbok/")
def arbok():
    return render_template("24-Arbok.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/pikachu/")
def pikachu():
    return render_template("25-Pikachu.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/raichu/")
def raichu():
    return render_template("26-Raichu.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/sandshrew/")
def sandshrew():
    return render_template("27-Sandshrew.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/sandslash/")
def sandslash():
    return render_template("28-Sandslash.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/nidoranF/")
def nidoranF():
    return render_template("29-NidoranF.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/nidorina/")
def nidorina():
    return render_template("30-Nidorina.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/nidoqueen/")
def nidoqueen():
    return render_template("31-Nidoqueen.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/nidoranM/")
def nidoranM():
    return render_template("32-NidoranM.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/nidorino/")
def nidorino():
    return render_template("33-Nidorino.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/nidoking/")
def nidoking():
    return render_template("34-Nidoking.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/clefairy/")
def clefairy():
    return render_template("35-Clefairy.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/clefable/")
def clefable():
    return render_template("36-Clefable.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/vulpix/")
def vulpix():
    return render_template("37-Vulpix.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/ninetales/")
def ninetales():
    return render_template("38-Ninetales.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/jigglypuff/")
def jigglypuff():
    return render_template("39-Jigglypuff.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/wigglytuff/")
def wigglytuff():
    return render_template("40-Wigglytuff.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/zubat/")
def zubat():
    return render_template("41-Zubat.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/golbat/")
def golbat():
    return render_template("42-Golbat.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/oddish/")
def oddish():
    return render_template("43-Oddish.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/gloom/")
def gloom():
    return render_template("44-Gloom.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])   

@app.route("/vileplume/")
def vileplume():
    return render_template("45-Vileplume.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/paras/")
def paras():
    return render_template("46-Paras.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/parasect/")
def parasect():
    return render_template("47-Parasect.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/venonat/")
def venonat():
    return render_template("48-Venonat.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/venomoth/")
def venomoth():
    return render_template("49-Venomoth.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/diglett/")
def diglett():
    return render_template("50-Diglett.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/dugtrio/")
def dugtrio():
    return render_template("51-Dugtrio.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/meowth/")
def meowth():
    return render_template("52-Meowth.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/persian/")
def persian():
    return render_template("53-Persian.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/psyduck/")
def psyduck():
    return render_template("54-Psyduck.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/golduck/")
def golduck():
    return render_template("55-Golduck.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/mankey/")
def mankey():
    return render_template("56-Mankey.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/primeape/")
def primeape():
    return render_template("57-Primeape.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/growlithe/")
def growlithe():
    return render_template("58-Growlithe.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/arcanine/")
def arcanine():
    return render_template("59-Arcanine.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/poliwag/")
def poliwag():
    return render_template("60-Poliwag.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/poliwhirl/")
def poliwhirl():
    return render_template("61-Poliwhirl.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/poliwrath/")
def poliwrath():
    return render_template("62-Poliwrath.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/abra/")
def abra():
    return render_template("63-Abra.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/kadabra/")
def kadabra():
    return render_template("64-Kadabra.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/alakazam/")
def alakazam():
    return render_template("65-Alakazam.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/machop/")
def machop():
    return render_template("66-Machop.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/machoke/")
def machoke():
    return render_template("67-Machoke.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/machamp/")
def machamp():
    return render_template("68-Machamp.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/bellsprout/")
def bellsprout():
    return render_template("69-Bellsprout.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/weepinbell/")
def weepinbell():
    return render_template("70-Weepinbell.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/victreebel/")
def victreebel():
    return render_template("71-Victreebel.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/tentacool/")
def tentacool():
    return render_template("72-Tentacool.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/tentacruel/")
def tentacruel():
    return render_template("73-Tentacruel.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/geodude/")
def geodude():
    return render_template("74-Geodude.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

@app.route("/graveler/")
def graveler():
    return render_template("75-Graveler.html", Pokemon1 = pokemonDict[request.args['number1']], Pokemon2 = pokemonDict[request.args['number2']],Pokemon3 = pokemonDict[request.args['number3']], Pokemon4 = pokemonDict[request.args['number4']], Pokemon5 = pokemonDict[request.args['number5']], first = request.args['number1'], second = request.args['number2'], third = request.args['number3'], forth = request.args['number4'], fifth = request.args['number5'], firstpercent = request.args['percent1'], secondpercent=request.args['percent2'], thirdpercent = request.args['percent3'], forthpercent = request.args['percent4'], fifthpercent = request.args['percent5'])

if __name__ == "__main__": 
    app.run()
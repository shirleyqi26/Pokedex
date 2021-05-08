from flask import Flask, redirect, url_for, render_template, request
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
  return pred

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
            pred = makePrediction(Image.open(img)).item()+1
            #REDIRECTIONS
            if pred == 1:
                return redirect(url_for("bulbasaur"))
            elif pred == 2:
                return redirect(url_for("ivysaur"))
            elif pred == 3:
                return redirect(url_for("venusaur"))
            elif pred == 4:
                return redirect(url_for("charmander"))
            elif pred == 5:
                return redirect(url_for("charmeleon"))
            elif pred == 6:
                return redirect(url_for("charizard"))
            elif pred == 7:
                return redirect(url_for("squirtle"))
            elif pred == 8:
                return redirect(url_for("wartortle"))
            elif pred == 9:
                return redirect(url_for("blastoise"))
            elif pred == 10:
                return redirect(url_for("caterpie"))
            elif pred == 11:
                return redirect(url_for("metapod"))
            elif pred == 12:
                return redirect(url_for("butterfree"))
            elif pred == 13:
                return redirect(url_for("weedle"))
            elif pred == 14:
                return redirect(url_for("kakuna"))
            elif pred == 15:
                return redirect(url_for("beedrill"))
            elif pred == 16:
                return redirect(url_for("pidgey"))
            elif pred == 17:
                return redirect(url_for("pidgeotto"))
            elif pred == 18:
                return redirect(url_for("pidgeot"))
            elif pred == 19:
                return redirect(url_for("rattata"))
            elif pred == 20:
                return redirect(url_for("raticate"))
            elif pred == 21:
                return redirect(url_for("spearow"))
            elif pred == 22:
                return redirect(url_for("fearow"))
            elif pred == 23:
                return redirect(url_for("ekans"))
            elif pred == 24:
                return redirect(url_for("arbok"))
            elif pred == 25:
                return redirect(url_for("pikachu"))
            elif pred == 26:
                return redirect(url_for("raichu"))
            elif pred == 27:
                return redirect(url_for("sandshrew"))
            elif pred == 28:
                return redirect(url_for("sandslash"))
            elif pred == 29:
                return redirect(url_for("nidoranF"))
            elif pred == 30:
                return redirect(url_for("nidorina"))
            elif pred == 31:
                return redirect(url_for("nidoqueen"))
            elif pred == 32:
                return redirect(url_for("nidoranM"))
            elif pred == 33:
                return redirect(url_for("nidorino"))
            elif pred == 34:
                return redirect(url_for("nidoking"))
            elif pred == 35:
                return redirect(url_for("clefairy"))
            elif pred == 36:
                return redirect(url_for("clefable"))
            elif pred == 37:
                return redirect(url_for("vulpix"))
            elif pred == 38:
                return redirect(url_for("ninetales"))
            elif pred == 39:
                return redirect(url_for("jigglypuff"))
            elif pred == 40:
                return redirect(url_for("wigglytuff"))
            elif pred == 41:
                return redirect(url_for("zubat"))
            elif pred == 42:
                return redirect(url_for("golbat"))
            elif pred == 43:
                return redirect(url_for("oddish"))
            elif pred == 44:
                return redirect(url_for("gloom"))
            elif pred == 45:
                return redirect(url_for("vileplume"))
            elif pred == 46:
                return redirect(url_for("paras"))
            elif pred == 47:
                return redirect(url_for("parasect"))
            elif pred == 48:
                return redirect(url_for("venonat"))                                                                                
            elif pred == 49:
                return redirect(url_for("venomoth"))
            elif pred == 50:
                return redirect(url_for("diglett"))
            elif pred == 51:
                return redirect(url_for("dugtrio"))
            elif pred == 52:
                return redirect(url_for("meowth"))
            elif pred == 53:
                return redirect(url_for("persian"))
            elif pred == 54:
                return redirect(url_for("psyduck"))
            elif pred == 55:
                return redirect(url_for("golduck"))
            elif pred == 56:
                return redirect(url_for("mankey"))
            elif pred == 57:
                return redirect(url_for("primeape"))
            elif pred == 58:
                return redirect(url_for("growlithe"))
            elif pred == 59:
                return redirect(url_for("arcanine"))
            elif pred == 60:
                return redirect(url_for("poliwag"))
            elif pred == 61:
                return redirect(url_for("poliwhirl"))
            elif pred == 62:
                return redirect(url_for("poliwrath"))
            elif pred == 63:
                return redirect(url_for("abra"))
            elif pred == 64:
                return redirect(url_for("kadabra"))
            elif pred == 65:
                return redirect(url_for("alakazam"))
            elif pred == 66:
                return redirect(url_for("machop"))
            elif pred == 67:
                return redirect(url_for("machoke"))
            elif pred == 68:
                return redirect(url_for("machamp"))
            elif pred == 69:
                return redirect(url_for("bellsprout"))
            elif pred == 70:
                return redirect(url_for("weepinbell"))
            elif pred == 71:
                return redirect(url_for("victreebel"))
            elif pred == 72:
                return redirect(url_for("tentacool"))
            elif pred == 73:
                return redirect(url_for("tentacruel"))
            elif pred == 74:
                return redirect(url_for("geodude"))
            elif pred == 75:
                return redirect(url_for("graveler"))
            else:
                return render_template("Home.html")
      else:
            return render_template("Home.html")
    else:
      return render_template("Home.html")
    
@app.route("/bulbasaur/")
def bulbasaur():
    return render_template("01-Bulbasaur.html")

@app.route("/ivysaur/")
def ivysaur():
    return render_template("02-Ivysaur.html")

@app.route("/venusaur/")
def venusaur():
    return render_template("03-Venusaur.html")

@app.route("/charmander/")
def charmander():
    return render_template("04-Charmander.html")

@app.route("/charmeleon/")
def charmeleon():
    return render_template("05-Charmeleon.html")

@app.route("/charizard/")
def charizard():
    return render_template("06-Charizard.html")

@app.route("/squirtle/")
def squirtle():
    return render_template("07-Squirtle.html")

@app.route("/wartortle/")
def wartortle():
    return render_template("08-Wartortle.html")

@app.route("/blastoise/")
def blastoise():
    return render_template("09-Blastoise.html")    

@app.route("/caterpie/")
def caterpie():
    return render_template("10-Caterpie.html")

@app.route("/metapod/")
def metapod():
    return render_template("11-Metapod.html")

@app.route("/butterfree/")
def butterfree():
    return render_template("12-Butterfree.html")

@app.route("/weedle/")
def weedle():
    return render_template("13-Weedle.html")

@app.route("/kakuna/")
def kakuna():
    return render_template("14-Kakuna.html")

@app.route("/beedrill/")
def beedrill():
    return render_template("15-Beedrill.html")

@app.route("/pidgey/")
def pidgey():
    return render_template("16-Pidgey.html")

@app.route("/pidgeotto/")
def pidgeotto():
    return render_template("17-Pidgeotto.html")

@app.route("/pidgeot/")
def pidgeot():
    return render_template("18-Pidgeot.html")

@app.route("/rattata/")
def rattata():
    return render_template("19-Rattata.html")

@app.route("/raticate/")
def raticate():
    return render_template("20-Raticate.html")    

@app.route("/spearow/")
def spearow():
    return render_template("21-Spearow.html")

@app.route("/fearow/")
def fearow():
    return render_template("22-Fearow.html")

@app.route("/ekans/")
def ekans():
    return render_template("23-Ekans.html")

@app.route("/arbok/")
def arbok():
    return render_template("24-Arbok.html")

@app.route("/pikachu/")
def pikachu():
    return render_template("25-Pikachu.html")

@app.route("/raichu/")
def raichu():
    return render_template("26-Raichu.html")

@app.route("/sandshrew/")
def sandshrew():
    return render_template("27-Sandshrew.html")

@app.route("/sandslash/")
def sandslash():
    return render_template("28-Sandslash.html")

@app.route("/nidoranF/")
def nidoranF():
    return render_template("29-NidoranF.html")

@app.route("/nidorina/")
def nidorina():
    return render_template("30-Nidorina.html")

@app.route("/nidoqueen/")
def nidoqueen():
    return render_template("31-Nidoqueen.html")

@app.route("/nidoranM/")
def nidoranM():
    return render_template("32-NidoranM.html")

@app.route("/nidorino/")
def nidorino():
    return render_template("33-Nidorino.html")

@app.route("/nidoking/")
def nidoking():
    return render_template("34-Nidoking.html")

@app.route("/clefairy/")
def clefairy():
    return render_template("35-Clefairy.html")

@app.route("/clefable/")
def clefable():
    return render_template("36-Clefable.html")

@app.route("/vulpix/")
def vulpix():
    return render_template("37-Vulpix.html")

@app.route("/ninetales/")
def ninetales():
    return render_template("38-Ninetales.html")

@app.route("/jigglypuff/")
def jigglypuff():
    return render_template("39-Jigglypuff.html")

@app.route("/wigglytuff/")
def wigglytuff():
    return render_template("40-Wigglytuff.html")

@app.route("/zubat/")
def zubat():
    return render_template("41-Zubat.html")

@app.route("/golbat/")
def golbat():
    return render_template("42-Golbat.html")

@app.route("/oddish/")
def oddish():
    return render_template("43-Oddish.html")

@app.route("/gloom/")
def gloom():
    return render_template("44-Gloom.html")    

@app.route("/vileplume/")
def vileplume():
    return render_template("45-Vileplume.html")

@app.route("/paras/")
def paras():
    return render_template("46-Paras.html")

@app.route("/parasect/")
def parasect():
    return render_template("47-Parasect.html")

@app.route("/venonat/")
def venonat():
    return render_template("48-Venonat.html")

@app.route("/venomoth/")
def venomoth():
    return render_template("49-Venomoth.html")

@app.route("/diglett/")
def diglett():
    return render_template("50-Diglett.html")

@app.route("/dugtrio/")
def dugtrio():
    return render_template("51-Dugtrio.html")

@app.route("/meowth/")
def meowth():
    return render_template("52-Meowth.html")

@app.route("/persian/")
def persian():
    return render_template("53-Persian.html")

@app.route("/psyduck/")
def psyduck():
    return render_template("54-Psyduck.html")

@app.route("/golduck/")
def golduck():
    return render_template("55-Golduck.html")

@app.route("/mankey/")
def mankey():
    return render_template("56-Mankey.html")

@app.route("/primeape/")
def primeape():
    return render_template("57-Primeape.html")

@app.route("/growlithe/")
def growlithe():
    return render_template("58-Growlithe.html")

@app.route("/arcanine/")
def arcanine():
    return render_template("59-Arcanine.html")

@app.route("/poliwag/")
def poliwag():
    return render_template("60-Poliwag.html")

@app.route("/poliwhirl/")
def poliwhirl():
    return render_template("61-Poliwhirl.html")

@app.route("/poliwrath/")
def poliwrath():
    return render_template("62-Poliwrath.html")

@app.route("/abra/")
def abra():
    return render_template("63-Abra.html")

@app.route("/kadabra/")
def kadabra():
    return render_template("64-Kadabra.html")

@app.route("/alakazam/")
def alakazam():
    return render_template("65-Alakazam.html")

@app.route("/machop/")
def machop():
    return render_template("66-Machop.html")

@app.route("/machoke/")
def machoke():
    return render_template("67-Machoke.html")

@app.route("/machamp/")
def machamp():
    return render_template("68-Machamp.html")

@app.route("/bellsprout/")
def bellsprout():
    return render_template("69-Bellsprout.html")

@app.route("/weepinbell/")
def weepinbell():
    return render_template("70-Weepinbell.html")

@app.route("/victreebel/")
def victreebel():
    return render_template("71-Victreebel.html")

@app.route("/tentacool/")
def tentacool():
    return render_template("72-Tentacool.html")

@app.route("/tentacruel/")
def tentacruel():
    return render_template("73-Tentacruel.html")


@app.route("/geodude/")
def geodude():
    return render_template("74-Geodude.html")

@app.route("/graveler/")
def graveler():
    return render_template("75-Graveler.html")

if __name__ == "__main__": 
    app.run()

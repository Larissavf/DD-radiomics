# DD-radiomics
##### Larissa Voshol | ##### 20-11-2023 | ##### Bio informatica 2 | ##### versie 1

## Introductie
In deze repo kan je de code en data vinden voor een data dashboard gemaakt in panel.
Het is een data dashboard waarin je kan kijken naar je pyradiomics dataset.
Dit is een dataset die je kan krijgen na het analyseren van een CT scan met behulp van Pyradiomics. Je krijgt een bestand dat is verdeeld in 8 groepen wat in totaal iets van 120 features heeft.
Dit zijn de verschillende groepen:
shape features:
beschrijft de 2D en 3D omvang en grootte van het ROI(region of interest). 

First order features:
beschrijft de distributie van de voxel intensiteit dat zich binnen in een bepaalde regio zitten.
    voxel: is a een waarde op een 3D grid, het zit niet zoals een pixel op vaste plek maar is afhankelijk van andere voxels voor de locatie. De voxel bevat dus alle intensiteit waarde tijdens de CT scan waar vervolgens de reconstructie op gebaseert wordt.

Gray Level Co-occurrence Matrix (GLCM) Features:
Is de creatie van een matrix over een afbeelding. En dat geeft de verdeling van de co-occuring pixel waardes in een grayscale. Vervolgens worden er naar verschillende aspecten hierin gekeken.

Gray Level Size Zone Matrix (GLSZM) Features:
Het kwantificeert de gray levels in verschillende zones in de afbeelding. En deze zones zijn gedefinieerd op basis van geconnected voxels op een afstand van 1 volgens de infinity norm.

Gray Level Run Length Matrix (GLRLM) Features:
Het kwantificeert de gray level runs, deze zijn gedefineerd als de aantal pixel dat de zelfde graylevels heeft.

Neighbouring Gray Tone Difference Matrix (NGTDM) Features:
Het kwantificeert de verschillen in gray levels van de gemiddelde waarde met die van zijn buur.

Gray Level Dependence Matrix (GLDM) Features:
Het kwantificeert de afhankelijkheid van een gray level in een afbeelding. De afhankelijkheid van een gray level wordt gedefineerd als de aantal verbonden voxels in een bepaalde afstand.

Op het dashboard kan je meerder patiënten tegelijk bekijken. Het is ook mogelijk om op het dashboard een klinisch bestand er ook bij te uploaden.
In dit klinische bestand kan je overige patiënten data in zetten. Wat mogelijk gerelateerd kan zijn aan 1 of meerdere Pyradiomics features.
**Let er wel op dat je de patiënten in beide bestanden op de zelfde volgorde heb staan, deze volgorde wordt gebruikt bij het maken van de grafieken**

Je kan de data bekijken met gebruik van een heatmap, om te kijken naar de correlatie tussen de features. Hierna kan je kijken naar de verspreiding in de features door het maken van een boxplot.
Dan kan je als laatst je features met elkaar vergelijken bij alle opgegeven patiënten. En extra filteren op een groep waarvan je de connectie van wil bekijken. Hierbij kan je dus 2 features tegenover elkaar zetten of 1 feauture tegen de patienten id's.

## benodigdheden
```
Linux debian 64 bit
Python 3
  -  panel v. 1.3.7
  -  pandas v. 2.2
  -  bokeh v. 3.3.3
  -  holoviews v. 1.18
  -  matplotlib v. 3.8
 
Windows 11
Python 3
  -  panel v. 1.3.7
  -  pandas v. 2.2
  -  bokeh v. 3.3.3
  -  holoviews v. 1.18
  -  matplotlib v. 3.8
```

## Project structuur
**Analyse**:
Bevat de exploratory data analysis.
**Dashboard**
Bevat de app.py bestand voor het runnen van het dashboard. Hiernaast bevat het de bestanden die gebruikt moeten worden voor het dashboard.
**Docs**
Bevat het hostingsplan.
**raw_data**
Bevat de orginele data die is verkegen bij het start van het project.
Alleen in deze bestanden waren de patiënten niet gelijk in beide bestanden, dit is aangepast in de bestanden die staan bij het dashboard.

## Gebruik
Om het dashboard te starten moet u beginnen met het clonen van de repo. Vervolgens naar Dashboard te gaan en daar de app.py te runnen.

**Om te starten:**
<img width="674" alt="image" src="https://github.com/Larissavf/DD-radiomics/assets/116642226/74c9f29a-a3cd-4329-877b-0914056a129d">
Begin met het uploaden van de goede bestanden in de aangewezen plekken. En vergeet niet de features te selecteren die je wilt vergelijken.

**volgende stap**
<img width="645" alt="image" src="https://github.com/Larissavf/DD-radiomics/assets/116642226/0b3d06b2-c0f8-4017-94d2-96862bc84f0b">
Dan komt deze knop te voorschijn als u die drukt, dan kunt u kiezen voor de grafiek die u wilt maken.
Hierbij kan je dus kiezen voor een Boxplot, Heatmap, Scatter(f1) en Scatter(f2)
Boxplot: kunt u de spreiding zien van de feautures in de z-score.
Heatmap: kunt u kijken naar de correlatie tussen de verschillende feautures.
Scatter(f1): kunt u 1 feauture vergelijken per patient verdeeld over een bepaalde groep.
Scatter(f2): kunt u 2 features tegenoverelkaar vergelijken met ook een bepaalde groep verdeling.

**grafiek**
<img width="478" alt="image" src="https://github.com/Larissavf/DD-radiomics/assets/116642226/6ca8e66e-6e4f-49a6-88c6-0c49d5e1fb7d">
Dan komt de grafiek in beeld, de knoppen die hier aanwezig zijn zijn afhankelijk van de soort grafiek die u wilt maken.
Maar hier kunt u de waardes aanpassen en zal de grafiek zich automatisch mee aanpassen.
U kunt ervoor kiezen om uw grafiek te verwijderen, zie de delete knop. Hij blijft jou keuzes onthouden. dus je kan er voor kiezen om hem later weer toe te voegen als u een andere volgorde in uw grafieken wil.

## Help
Als er hulp nodig is of er vragen zijn:
Larissa Voshol: l.voshol@st.hanze.nl






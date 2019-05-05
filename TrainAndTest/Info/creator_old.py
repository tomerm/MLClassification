import os
import glob
import json
import collections
import webbrowser
from datetime import date, timedelta
from Utils.utils import fullPath

class InfoCreator:
    def __init__(self, Config):
        print ("Start to create info...")
        self.Config = Config
        self.curDir = os.path.dirname(__file__)
        self.info = {}
        self.startId = "%d%0.2d%0.2d000000"%(date.today().year, date.today().month, date.today().day)
        if self.Config["infofrom"] != "today":
            arr = self.Config["infofrom"].split()
            prevDays = int(arr[0])
            startDay = date.today() - timedelta(days=prevDays)
            self.startId = "%d%0.2d%0.2d000000" % (startDay.year, startDay.month, startDay.day)
        self.path = fullPath(Config, "reportspath")
        os.chdir(self.path)
        for f in glob.glob("*"):
            resPath = self.path + "/" + f
            try:
                ind = f.rindex(".")
            except ValueError:
                ind = len(f)
            key = f[:ind]
            if (key < self.startId):
                continue
            with open(resPath, 'r', encoding='utf-8') as json_file:
                try:
                    self.info[key] = json.load(json_file)
                except json.JSONDecodeError:
                    print ("Warning: file %s doesn't have json format. Skipped."%(resPath))
            json_file.close()
        if len(self.info) == 0:
            print ("Folder %s doesn't contain reports, created in required diapason of dates. Exit."%(self.path))
            return
        self.html = ""
        self.qReqs = 0
        self.footer = "</table></body></html>"
        self.docsDict = self.getDocsDictionary()
        self.createHtml()

    def createHtml(self):
        self.html = "<!DOCTYPE html><html><head><meta charset='utf-8' />"
        self.html += "<link rel='stylesheet' type='text/css' href='%s/styles.css'>"%(self.curDir)
        self.html += "<script type='text/javascript'>var fullInfo='%s'</script>" % (json.dumps(self.info))
        self.html += "<script type='text/javascript'>var docsInfo='%s'</script>" % (json.dumps(self.docsDict))
        self.html += "<script type='text/javascript' src='%s/scripts.js'></script>"%(self.curDir)
        self.html += "</head><body>"
        self.html += "<h2 style='text-align:center'>Compare models</h2>"
        self.html += "<hr style='height: 1px; width: 100%'>"
        self.html += "<table style='width:100%; height:95%; position: relative; top: -30px;'>"
        self.html += "<tr><td style='width:8%; min-width: 200px; height: 90%'>"
        self.html += "<div style='height: 100%; border-right: 2px solid black;'>"
        self.html += "<h4 style='text-align: center; height: 20px; position:relative; bottom: -10px'>Requests</h4>"
        self.html += "<div style='width: 100%'; height: 20px; text-align: center;'>"
        self.html += "<button type='button' value='select' id='bselect' onclick='addAllRequests(this)' "
        self.html += "style='width: 90%; text-align:center'>Select All</button></div><p></p>"
        self.html += "<div style='overflow: auto; width: 100%; height: 100%;'>"
        self.html += "<table id='reqlist' style='width:100%'>"
        for key, val in self.info.items():
            self.html += "<tr style='height: 20px'><td style='width:15%'><input type='checkbox' class='chk' "
            self.html += "onclick='addRequest(this, %s)' "%(key)
            self.html += "id='chk_" + key + "' /></td>"
            self.html += "<td style='text-align: center' >" + key + "</td></tr>"
        self.html += "</table></div></div></td>"
        self.html += "<td id='tdPage' style='height: 100%; min-height: 100%;'>" + self.createMainPage() + "</td></tr>"
        self.html += self.footer
        path = self.path + "/curInfo.html"
        with open(path, 'w', encoding='utf-8') as file:
            file.write(self.html)
        file.close()
        print ("Launch report...")
        webbrowser.open(path)

    def createMainPage(self):
        pages = ["Requests", "Models", "Categories", "Documents"]
        mainHtml = "<div style='width:100%; height: 100%; min-height: 100%;'>"
        mainHtml += "<ul class='menu' style='border-bottom: 2px solid black;'>"
        for i in range(len(pages)):
            mainHtml += "<li class='menu' id='%s' onclick='changePage(\"%s\")'>%s</li>"%("page" + pages[i], pages[i], pages[i])
        mainHtml += "</ul><div id='mainPage' style='height:100%; min-height:100%; width: 100%; overflow: auto;'>"
        mainHtml += "</div>"
        return mainHtml

    def getDocsDictionary(self):
        print ("Read source files...")
        docs = collections.OrderedDict()
        fcont = ""
        for key, val in self.info.items():
            path = self.Config["home"] + "/" + self.info[key]["sourcesPath"]
            for pathf, subdirs, files in os.walk(path):
                for doc in files:
                    if doc not in docs:
                        if not os.path.isfile(os.path.join(pathf, doc)):
                            print ("File %s doesn't exist."%(path+doc))
                            continue
                    #with open(os.path.join(pathf, doc), "r", encoding="utf-8") as file:
                    #    fcont = file.read()
                    #file.close()
                    docs[doc] = os.path.join(pathf, doc)

        return collections.OrderedDict(sorted(docs.items()))

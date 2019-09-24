import os
import glob
import json
import collections
import webbrowser
import logging
from datetime import date, timedelta
from Utils.utils import get_abs_path
import General.settings as settings

logger = logging.getLogger(__name__)

class InfoCreator:
    def __init__(self):
        logger.info("Start to create info...")
        self.curDir = os.path.dirname(__file__)
        self.info = {}
        self.startId = "%d%0.2d%0.2d000000"%(date.today().year, date.today().month, date.today().day)
        if settings.Config["info_from"] != "today":
            arr = settings.Config["info_from"].split()
            prev_days = int(arr[0])
            start_day = date.today() - timedelta(days=prev_days)
            self.startId = "%d%0.2d%0.2d000000" % (start_day.year, start_day.month, start_day.day)
        self.path = get_abs_path(settings.Config, "reports_path")
        os.chdir(self.path)
        for f in glob.glob("*"):
            res_path = self.path + "/" + f
            try:
                ind = f.rindex(".")
            except ValueError:
                ind = len(f)
            key = f[:ind]
            if (key < self.startId):
                continue
            with open(res_path, 'r', encoding='utf-8') as json_file:
                try:
                    self.info[key] = json.load(json_file)
                except json.JSONDecodeError:
                    logger.warning("Warning: file %s doesn't have json format. Skipped." % res_path)
            json_file.close()
        if not self.info:
            logger.error("Folder %s doesn't contain reports, created in required diapason of dates. Exit." % self.path)
            return
        self.html = ""
        self.qReqs = 0
        self.footer = "</table></body></html>"
        self.docsDict = self.get_docs_dictionary()
        self.create_html()

    def get_css_and_js(self):
        css = ""
        funcjs = ""
        with open(self.curDir + "/scripts.js", 'r', encoding='UTF-8') as tc:
            for line in tc:
                funcjs += line
        tc.close()
        with open(self.curDir + "/styles.css", 'r', encoding='UTF-8') as tc:
            for line in tc:
                css += line
        tc.close()
        return funcjs, css

    def create_html(self):
        fjs, css = self.get_css_and_js()
        self.html = "<!DOCTYPE html><html><head><meta charset='utf-8' />"
        self.html += "<style>" + css + "</style>"
        #self.html += "<link rel='stylesheet' type='text/css' href='%s/styles.css'>"%(self.curDir)
        self.html += "<script type='text/javascript'>var fullInfo='%s'</script>" % (json.dumps(self.info))
        self.html += "<script type='text/javascript'>var docsInfo='%s'</script>" % (json.dumps(self.docsDict))
        #self.html += "<script type='text/javascript' src='%s/scripts.js'></script>"%(self.curDir)
        self.html += "<script type='text/javascript'>" + fjs + "</script>"
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
            self.html += "onclick='addRequest(this, %s)' " % (key)
            self.html += "id='chk_" + key + "' /></td>"
            self.html += "<td style='text-align: center' >" + key + "</td></tr>"
        self.html += "</table></div></div></td>"
        self.html += "<td id='tdPage' style='height: 100%; min-height: 100%;'>" + self.create_main_page() + "</td></tr>"
        self.html += self.footer
        path = self.path + "/curInfo.html"
        with open(path, 'w', encoding='utf-8') as file:
            file.write(self.html)
        file.close()
        logger.info("Launch report...")
        webbrowser.open(path)

    def create_main_page(self):
        pages = ["Requests", "Models", "Categories", "Documents"]
        main_html = "<div style='width:100%; height: 100%; min-height: 100%;'>"
        main_html += "<ul class='menu' style='border-bottom: 2px solid black;'>"
        for p in pages:
            main_html += "<li class='menu' id='%s' onclick='changePage(\"%s\")'>%s</li>" % ("page" + p, p, p)
        main_html += "</ul><div id='mainPage' style='height:100%; min-height:100%; width: 100%; overflow: auto;'>"
        main_html += "</div>"
        return main_html

    def get_docs_dictionary(self):
        logger.info("Read source files...")
        docs = collections.OrderedDict()
        for key, val in self.info.items():
            path = settings.Config["home"] + "/" + self.info[key]["sourcesPath"]
            for pathf, subdirs, files in os.walk(path):
                for doc in files:
                    if doc not in docs:
                        if not os.path.isfile(os.path.join(pathf, doc)):
                            logger.warning("File %s doesn't exist."%(path+doc))
                            continue
                    #with open(os.path.join(pathf, doc), "r", encoding="utf-8") as file:
                    #    fcont = file.read()
                    #file.close()
                    docs[doc] = os.path.join(pathf, doc)

        return collections.OrderedDict(sorted(docs.items()))

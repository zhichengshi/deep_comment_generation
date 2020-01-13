import json
import _pickle as pickle
import os
import subprocess
import xml.etree.ElementTree as ET
import json
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import pandas as pd

class pipeline:

    def __init__(self, srcJsonPath, xmlCommentPath):
        self.srcJsonPath = srcJsonPath
        self.xmlCommentPath = xmlCommentPath

    def check_or_create(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

    def pair_src_comment(self):
        src = []
        comments = []
        file_name = self.srcJsonPath.split(os.sep)[-1].split(".")[0]
        root = os.path.dirname(self.srcJsonPath)
        write_dir = os.path.join(root, file_name)
        self.check_or_create(write_dir)

        f = open(self.srcJsonPath, encoding="utf-8")
        for line in f:
            code = json.loads(line)["code"]
            comment = json.loads(line)["nl"]
            src.append(code)
            comments.append(comment)

        # batch pair code and comment
        assert len(comments) == len(src)
        start = 0
        step = 1000
        while start + step < len(src):
            path = os.path.join(write_dir, str(
                start) + "_"+str(start+step)+".java")
            with open(path, "w") as f:
                for i in range(start, start+step):
                    comment_src = "//"+comments[i]+"\n"+src[i]
                    f.writelines(comment_src)
            start += step

        path = os.path.join(write_dir, str(start)+"_"+str(len(src))+".java")
        with open(path, "w") as f:
             for i in range(start, len(src)):
                comment_src = "//"+comments[i]+"\n"+src[i]
                f.writelines(comment_src)

        # generate xml
        xml_dir = os.path.join(root, "xml")
        xml_path = os.path.join(xml_dir, file_name+".xml")
        self.check_or_create(xml_dir)

        cmd = "srcml.exe " + write_dir + " -o " + xml_path
        subprocess.check_output(cmd)

    # construct xml and comment pair
    def xmlComment(self):
        def lemmatize(comment):
            # get the word part of speech
            def get_wordnet_pos(tag):
                if tag.startswith('J'):
                    return wordnet.ADJ
                elif tag.startswith('V'):
                    return wordnet.VERB
                elif tag.startswith('N'):
                    return wordnet.NOUN
                elif tag.startswith('R'):
                    return wordnet.ADV
                else:
                    return None
            
            tokens=word_tokenize(comment) # split tokens
            tagged_sent=pos_tag(tokens) # get teh word part of speech

            wnl = WordNetLemmatizer()
            lemmas_sent = []
            for tag in tagged_sent:
                wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN 
                
                lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos)) # lemmatize

            comment=' '.join(lemmas_sent)
            return comment
        

        file_name = self.srcJsonPath.split(
            os.sep)[-1].split(".")[0]+".xml"  # train validate or test
        xml_path = os.path.join(os.path.dirname(
            self.srcJsonPath), "xml", file_name)
        dom_tree = ET.parse(xml_path).getroot()

        body_name = set(["function", "constructor"])
        comments = []
        code = []
        for unit in dom_tree.getchildren():  # every unit represent a file
            for i in range(len(unit)):
                if i+1 < len(unit) and unit[i].tag == "comment" and unit[i+1].tag in body_name:
                    comments.append(unit[i])
                    code.append(unit[i+1])

        assert len(comments) == len(code)
        dirname = os.path.dirname(self.xmlCommentPath)
        self.check_or_create(dirname)
        for i in range(len(comments)):
            # represent the ast in xml form
            code[i] = ET.tostring(
                code[i], encoding='utf-8').decode('utf-8')
            
            # preprocess comment
            comment_text=comments[i].text.lstrip('/')
            comments[i]=lemmatize(comment_text)
       
       # dump to csv
        df=pd.DataFrame()
        df['code']=code
        df['comment']=comments
        df.to_csv(self.xmlCommentPath)

        return 

        

    

if __name__ == "__main__":
    srcJsonPath = "C:\\Users\\w\\Desktop\\temp\\train.json"
    xmlCommentJsonPath = "C:\\Users\\w\\Desktop\\temp\\xmlComment\\train.json"
    ppl = pipeline(srcJsonPath, xmlCommentJsonPath)
    ppl.xmlComment()

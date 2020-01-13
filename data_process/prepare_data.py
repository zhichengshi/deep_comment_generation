import xml.etree.ElementTree as ET
import re



def split_token(token):  # split token
    pattern = re.compile(r'[a-z]+|[A-Z]{1}[a-z]*')
    subTokens = pattern.findall(token)
    return subTokens



def get_ast_sequences(root, sequence):  # get the ast node sequence
    if root.tail != None:
        sequence.append(root.tag)
        if root.text != None:
            sequence += split_token(root.text)
        for node in root:
            get_ast_sequences(node, sequence)
        sequence.append(root.tail)
    else:
        sequence.append(root.tag)
        if root.text != None:
            sequence += split_token(root.text)
        for node in root:
            get_ast_sequences(node, sequence)
    


def get_blocks(root,blocks):
    # add parent into ast node object
    class treeNode:
        def __init__(self, parent, ele):
            if parent != None:
                self.parent = parent
                self.ele = ele
            else:
                self.parent = parent
                self.ele = ele

    # leverage leaves when traversing the ast 
    def transform(root):
        if root.text != None:
            root.append(ET.Element(root.text))
        for child in root:
            transform(child)
        return root

    # deep first traverse
    def dfs(root, list, parent=None):
        list.append(treeNode(parent, root))
        for node in root:
            dfs(node, list, root)
        return list

    split_nodes=set(["decl_stmt","expr_stmt","function","constructor"])
    dfs_node_sequence=dfs(root,[],None)
    for node in dfs_node_sequence:
        if node.ele.tag in split_nodes:
            blocks.append(node.ele)
            if node.parent != None:
                node.parent.remove(node.ele)\
                    
    blocks=[transform(block) for block in blocks]
    return blocks

  





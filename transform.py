import pandas as pd
import ast
import numpy as np

# Load the CSV file
def getOption(image,green_intersection,marked_positions,red_margin_x):
    options=["a","b","c","d"]

    grid_height=220
    grid_gap=85
    row_width=300
    circle_width=60


    grid_x=green_intersection[0]
    grid_y=green_intersection[1]-grid_height

    result={"set":None,"data":[]}

    for marked_position in marked_positions:
        
        grid="left"
        if(marked_position[1]<250):
            grid="set"
            text_width=146 if "pretest" in image else 180
            rel_marked_position_x=marked_position[0]-red_margin_x-text_width
            marked_col=rel_marked_position_x/(246/5)
            result["set"]=int(marked_col+1)
            continue
            
            


        
        
        
    
        

        rel_marked_position_x=marked_position[0]-grid_x
        rel_marked_position_y=marked_position[1]-grid_y

        if(rel_marked_position_x>row_width+10):
            grid="right"
            rel_marked_position_x-=grid_gap+row_width
            
        
        marked_col=rel_marked_position_x/(row_width/4)
        marked_row=rel_marked_position_y/(grid_height/3)
        print(marked_row)

        left_grid_question_no=[1,3,5]
        right_grid_question_no=[2,4]
        
        question_no=left_grid_question_no[int(marked_row)] if grid=="left" else right_grid_question_no[int(marked_row)]
            
                
        result["data"].append([question_no,options[int(marked_col)]])
    return result

    
    






# print(getOption((150,2288),[(181, 2101), (341, 2177), (423, 2259), (392, 111), (820, 2177), (741, 2099)],85))

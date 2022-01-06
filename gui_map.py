from tkinter import * 
from tkinter import ttk
import numpy as np
from PIL import Image

class GameWindow:
    '''This is a class to represent the map for frozen lake game while agent is playing
    '''
    def __init__(self,parent,gmap):
        '''
            Args:
            
                gmap: 2D array of strings for map layout
                parent: TK parent element
        '''
        self.cellsize = 150
        self.scale = 3
        self.map = gmap
        self.dim = len(self.map)
        self.ws = parent
        self.ws.title("game map")
        self.ws.geometry(str(self.cellsize*len(self.map))+"x"+str(self.cellsize*len(self.map)))
        self.ws['bg'] = '#AC99F2'
        self.container = Frame(self.ws)
        self.container.pack()
        
        self.canvas = Canvas(self.container,
                                     width= self.cellsize * self.dim,
                                     height= self.cellsize * self.dim) 
        self.canvas.grid()
        self.__configmap()   
    def __configmap(self):
        '''creates the map
        '''
        self.color = ''
        for row in range(len(self.map)):
            for column in range(len(self.map)):
                if self.map[row][column] == '.':
                    self.color = '#0f9af7'
                elif self.map[row][column] == '&':
                    self.color = '#8b9094'
                elif self.map[row][column] == '#':
                    self.color = '#054375'
                elif self.map[row][column] == '$':
                    self.color = '#ffffff'
                self.canvas.create_rectangle( self.cellsize* column,
                                              self.cellsize* row,
                                              self.cellsize* (column+1),
                                              self.cellsize*(row+1),
                                              fill = self.color)
                
    def __create_arrow_map(self,arrows):
        '''creates arrow map
            Args:
                arrows(list of lists containing arrow symbols): a 2D array of arrows for each cell in the map
        '''
        for row in range(len(self.map)):
            for column in range(len(self.map)):
                if arrows[row][column] == 0:
                    self.canvas.create_line(self.cellsize*column + 22.5,
                                            self.cellsize*row+ 22.5,
                                            self.cellsize*column + 22.5,
                                            self.cellsize*row + 22.5+30*self.scale,arrow = FIRST)
                elif arrows[row][column] == 1:
                    self.canvas.create_line(self.cellsize*column + 22.5,
                                            self.cellsize*row+ 22.5,
                                            self.cellsize*column + 22.5,
                                            self.cellsize*row + 22.5+30*self.scale,arrow = LAST)
                elif arrows[row][column] == 2:
                    self.canvas.create_line(self.cellsize*column + 22.5,
                                            self.cellsize*row+ 22.5,
                                            self.cellsize*column + 22.5+30*self.scale,
                                            self.cellsize*row + 22.5,arrow = FIRST)
                elif arrows[row][column] == 3:
                    self.canvas.create_line(self.cellsize*column + 22.5,
                                            self.cellsize*row+ 22.5,
                                            self.cellsize*column + 22.5+30*self.scale,
                                            self.cellsize*row + 22.5,arrow = LAST)
    def __create_value_map(self,values):
        '''creates value map
            Args:
                values(list of lists or np array): a 2D array of values for each cell in the map
        '''
        for row in range(len(self.map)):
            for column in range(len(self.map)):
                self.canvas.create_text(self.cellsize * column+ 50+self.cellsize/self.scale,
                                        self.cellsize * row+ 50+self.cellsize/self.scale,
                                        text=str(np.round(values[row][column],2)),
                                        fill="black",
                                        font=('Helvetica '+str(10*self.scale)+' bold'))
    def __create_image(self, mapname):
        self.canvas
        
        self.canvas.postscript(height=self.cellsize*len(self.map),
                               width=self.cellsize*len(self.map),
                               file=mapname+".eps")
        img = Image.open(mapname+'.eps')
        img.save(mapname+'.jpg')
        img.show() 
    def create_arrow_value_map(self,gmap ,values,arrows,mapname):
        '''use this method to create the map with policy and value overlays
            Args:
                gmap: 2D array of strings for map layout
                values(list of lists or np array): a 2D array of values for each cell in the map
                arrows(list of lists containing arrow symbols): a 2D array of arrows for each cell in the map
        '''    
        # setting the layout
        self.map = gmap
        self.__configmap()
        # creating arrow map
        self.__create_arrow_map(arrows)
        #creating value map
        self.__create_value_map(values)
        #creating image for later use
        self.__create_image(mapname)
    
    def prepare_data(self,val,arr):
        valuemap = np.delete(val,[-1]).reshape(len(self.map),len(self.map))
        policymap = np.delete(arr,[-1]).reshape(len(self.map),len(self.map))
        return valuemap,policymap
    def update_window(self,gmap):
        '''updates map layout for screenrecording 
            Args : 
                gmap: 2D array of strings for map layout
        '''
        self.map = gmap
        self.__configmap()
    

## This is how we use this file make sure we delete it later :)    
    
# lake =  [['&', '.', '.', '.'],
#          ['.', '#', '.', '#'],
#          ['.', '.', '.', '#'],
#          ['#', '.', '.', '$']]

    
# lake2 =  [['&', '.', '.', '.'],
#          ['#', '#', '.', '#'],
#          ['#', '#', '.', '#'],
#          ['#', '.', '.', '$']]

# Policy=[['↓', '→', '↓', '←'],
#         ['↓', '↑', '↓', '↑'],
#         ['→', '↓', '↓', '↑'],
#         ['↑', '→', '→', '↑']]

# value = [[0.455, 0.504, 0.579, 0.505],
#          [0.508, 0.,    0.653, 0.   ],
#          [0.584, 0.672, 0.768, 0.   ],
#          [0.,    0.771, 0.887, 1.   ],]
# root = Tk()
# g = GameWindow(root,lake)

# g.create_arrow_value_map(lake,value,Policy)
# root.mainloop()

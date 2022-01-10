from tkinter import * 
import numpy as np
from PIL import Image

'''GameWindow class
    This is a class to draw the map of the frozen lake inlcuding the policy and the values as a image
'''    
class GameWindow:
    ''' __init__ function

        @param gmap
            2D array of strings for map layout
        @param parent
            TK parent element
    '''
    def __init__(self,parent,gmap):
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


    '''__configmap function
        creates the map
    '''
    def __configmap(self):
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
    
    '''__create_arrow_map function
        addes the arrows representing the policy to the map
        
        @param arrows
            list of lists containing arrow symbols: a 2D array of arrows for each cell in the map
    '''          
    def __create_arrow_map(self,arrows):
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


    '''__create_value_map function
        added the values to the map

        @param values
            list of lists or np array: a 2D array of values for each cell in the map
    '''
    def __create_value_map(self,values):
        for row in range(len(self.map)):
            for column in range(len(self.map)):
                self.canvas.create_text(self.cellsize * column+ 50+self.cellsize/self.scale,
                                        self.cellsize * row+ 50+self.cellsize/self.scale,
                                        text=str(np.round(values[row][column],2)),
                                        fill="black",
                                        font=('Helvetica '+str(10*self.scale)+' bold'))

    ''' __create_image function
        make an image out of the created map and safe the image as a .jpg

        @param mapname
            the name of the saved image
    '''
    def __create_image(self, mapname):
        self.canvas
        
        self.canvas.postscript(height=self.cellsize*len(self.map),
                               width=self.cellsize*len(self.map),
                               file=mapname+".eps")
        img = Image.open(mapname+'.eps')
        img.save(mapname+'.jpg')
        img.show() 

    ''' prepare_data function
        removed the absorbing states from the policy and the values, as they are not represented within the map

        @param val
            the values
        @param arr
            the policy
        @return valuemap
            the values without the value for the absorbing state
        @return policymap
            the policy without the policy for the absorbing state
    '''
    def __prepare_data(self,val,arr):
        valuemap = np.delete(val,[-1]).reshape(len(self.map),len(self.map))
        policymap = np.delete(arr,[-1]).reshape(len(self.map),len(self.map))
        return valuemap,policymap

    ''' create_arrow_value_map function
        use this method to create the map with policy and value overlays
        
        @param gmap
            2D array of strings for map layout
        @param values
            list of lists or np array: a 2D array of values for each cell in the map
        @param arrows
            list of lists containing arrow symbols: a 2D array of arrows for each cell in the map
    '''  
    def create_arrow_value_map(self,gmap ,values,arrows,mapname):  
        values, arrows = self.__prepare_data(values, arrows)
        self.map = gmap
        self.__configmap()
        self.__create_arrow_map(arrows)
        self.__create_value_map(values)
        self.__create_image(mapname)
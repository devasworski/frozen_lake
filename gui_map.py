from tkinter import * 
from tkinter import ttk


class GameWindow:
    '''This is a class to represent the map for frozen lake game while agent is playing
    '''
    def __init__(self,parent,gmap):
        '''
            Args:
            
                gmap: 2D array of strings for map layout
                parent: TK parent element
        '''
        
        self.map = gmap
        self.dim = len(self.map)
        self.ws = parent
        self.ws.title("game map")
        # self.ws.geometry("500*500")
        self.ws['bg'] = '#AC99F2'
        self.container = Frame(self.ws)
        self.container.pack()
        
        self.canvas = Canvas(self.container,
                                     width= 75 * self.dim,
                                     height= 75 * self.dim) 
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
                self.canvas.create_rectangle( 75* column,
                                              75* row,
                                              75* (column+1),
                                              75*(row+1),
                                              fill = self.color)
                
    def __create_arrow_map(self,arrows):
        '''creates arrow map
            Args:
                arrows(list of lists containing arrow symbols): a 2D array of arrows for each cell in the map
        '''
        for row in range(len(self.map)):
            for column in range(len(self.map)):
                if arrows[row][column] == '↑':
                    self.canvas.create_line(75*column + 22.5,
                                            75*row+ 22.5,
                                            75*column + 22.5,
                                            75*row + 22.5+30,arrow = FIRST)
                elif arrows[row][column] == '↓':
                    self.canvas.create_line(75*column + 22.5,
                                            75*row+ 22.5,
                                            75*column + 22.5,
                                            75*row + 22.5+30,arrow = LAST)
                elif arrows[row][column] == '←':
                    self.canvas.create_line(75*column + 22.5,
                                            75*row+ 22.5,
                                            75*column + 22.5+30,
                                            75*row + 22.5,arrow = FIRST)
                elif arrows[row][column] == '→':
                    self.canvas.create_line(75*column + 22.5,
                                            75*row+ 22.5,
                                            75*column + 22.5+30,
                                            75*row + 22.5,arrow = LAST)
    def __create_value_map(self,values):
        '''creates value map
            Args:
                values(list of lists or np array): a 2D array of values for each cell in the map
        '''
        for row in range(len(self.map)):
            for column in range(len(self.map)):
                self.canvas.create_text(75 * column+ 50,
                                        75 * row+ 50,
                                        text=str(values[row][column]),
                                        fill="black",
                                        font=('Helvetica 10 bold'))
        
    def create_arrow_value_map(self,gmap ,values,arrows):
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
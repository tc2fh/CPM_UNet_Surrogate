
from cc3d.core.PySteppables import *
from cc3d import CompuCellSetup
from cc3d.CompuCellSetup import persistent_globals as pg
import numpy as np
import zarr


PythonCall = 0 #set to 1 if running from python, 0 if running from CC3D

class AngiogenesisSteppable(SteppableBasePy):

    def __init__(self,frequency=1):

        SteppableBasePy.__init__(self,frequency)

    def start(self):
        """
        any code in the start function runs before MCS=0
        """

        print('number of cells = ',len(self.cell_list_by_type(self.EC)))

        if PythonCall:

            values2pass = pg.input_object
            global RunNumber
            global OutputPath
           
            RunNumber = values2pass[0] #run number for naming file
            OutputPath = values2pass[1] #path to folder where output zarr file will be saved


            #create zarr file with zipstore backend
            zarrpath = OutputPath + f'/simulation_{RunNumber}.zarr.zip'
            self.store = zarr.ZipStore(zarrpath, mode='w')
            self.root = zarr.group(store=self.store)

            #pre-allocate arrays for cell_id, fgbg, and vegf
            #comment out  self.cell_id_array because we decided not to train on instance segmentation
            # self.cell_id_array = self.root.zeros("cell_id", shape=(2000, self.dim.y, self.dim.x), chunks=(1, self.dim.y, self.dim.x), dtype=int)
            self.fgbg_array = self.root.zeros("fgbg", shape=(20000, self.dim.y, self.dim.x), chunks=(1, self.dim.y, self.dim.x), dtype=int)
            self.vegf_array = self.root.zeros("vegf", shape=(20000, self.dim.y, self.dim.x), chunks=(1, self.dim.y, self.dim.x), dtype=float)

    def step(self,mcs):


        

        if PythonCall:
            #write to zarr with zipstore backend
            VEGF = self.field.VEGF
            shape = (self.dim.y,self.dim.x) #images are in z,y,x convention, or depth, height, width
            
            cell_id_data = np.zeros(shape,dtype=np.uint16)
            vegf_data = np.zeros(shape,dtype=np.float32)
            for x, y, z in self.every_pixel():
                vegf_data[y,x] = VEGF[x,y,z]
                cell = self.cell_field[x, y, z]
                if cell: cell_id_data[y,x] = cell.id
            
            fgbg_data = np.where(cell_id_data != 0, 1, 0)

            # Write data to the pre-allocated arrays in the zarr file
            # self.cell_id_array[mcs] = cell_id_data #commented out to prevent saving instance segmentation
            self.fgbg_array[mcs] = fgbg_data
            self.vegf_array[mcs] = vegf_data
        

        #stop sim at mcs = 300 for testing
        # if mcs == 300:
        #     self.stop_simulation()

    def finish(self):
        """
        Finish Function is called after the last MCS
        """
        if PythonCall:
            self.store.close()
        

    def on_stop(self):
        # this gets called each time user stops simulation
        if PythonCall:
            self.store.close()
        return


        
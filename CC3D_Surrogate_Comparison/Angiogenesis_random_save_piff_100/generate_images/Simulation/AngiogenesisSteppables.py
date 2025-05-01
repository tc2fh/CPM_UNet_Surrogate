from cc3d.core.PySteppables import *
from cc3d import CompuCellSetup
from cc3d.CompuCellSetup import persistent_globals as pg
import numpy as np
import zarr
import os

SaveFiles = 1 #set to 1 if auto saving files every 100 mcs, 0 otherwise

class AngiogenesisSteppable(SteppableBasePy):

    def __init__(self,frequency=1):

        SteppableBasePy.__init__(self,frequency)

    def start(self):
        """
        any code in the start function runs before MCS=0
        """

    def step(self,mcs):

        if SaveFiles:
            if mcs % 100 == 0:
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

                #save as zarr zipstore with fgbg_data in group 'fgbg', vegf_data in group 'vegf'
                #create zarr file with zipstore backend
                zarr_name = f'mcs_{mcs}.zarr.zip'
                zarrpath = os.path.join(os.path.dirname(__file__), 'zarr_files', zarr_name)
                os.makedirs(os.path.dirname(zarrpath), exist_ok=True)
                zarr_store = zarr.ZipStore(zarrpath, mode='w')
                zarr_root = zarr.group(store=zarr_store)

                #pre-allocate arrays for cell_id, fgbg, and vegf
                #comment out  self.cell_id_array because we decided not to train on instance segmentation
                # self.cell_id_array = zarr_root.zeros("cell_id", shape=(2000, self.dim.y, self.dim.x), chunks=(1, self.dim.y, self.dim.x), dtype=int)
                fgbg_array = zarr_root.zeros("fgbg", shape=(self.dim.y, self.dim.x), chunks=(self.dim.y,self.dim.x), dtype=int)
                vegf_array = zarr_root.zeros("vegf", shape=(self.dim.y, self.dim.x), chunks=(self.dim.y,self.dim.x), dtype=float)

                # Write data to the pre-allocated arrays in the zarr file
                # self.cell_id_array[mcs] = cell_id_data #commented out to prevent saving instance segmentation
                fgbg_array[:] = fgbg_data
                vegf_array[:] = vegf_data

                #close the zarr file
                zarr_store.close()
                print(f'zarr file saved at mcs {mcs}')
            
    def finish(self):
        """
        Finish Function is called after the last MCS
        """
        
    def on_stop(self):
        # this gets called each time user stops simulation
        # self.IMAGE.close()
        return


        
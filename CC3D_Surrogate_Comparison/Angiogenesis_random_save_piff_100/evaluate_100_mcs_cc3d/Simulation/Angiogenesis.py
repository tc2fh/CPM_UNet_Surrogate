
from cc3d import CompuCellSetup
        

from AngiogenesisSteppables import AngiogenesisSteppable

CompuCellSetup.register_steppable(steppable=AngiogenesisSteppable(frequency=1))


CompuCellSetup.run()

"""
@Problem: 2D Cook's membrane. A benchmark example.

@Formulation: Mixed displacement-pressure formulation.

@Material: Incompressible Neo-Hookean.

@author: Dr Chennakesava Kadapa

Created on Sun 18-Jun-2024
"""


import dolfinx
import numpy as np

# For MPI-based parallelization
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from petsc4py import PETSc

# specific functions from dolfinx modules
from dolfinx import fem, mesh, io, plot, log
from dolfinx.fem import (Constant, dirichletbc, Function, functionspace, Expression, form, assemble_scalar )
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile
#from dolfinx.io import VTXWriter

import ufl
from ufl import (TestFunctions, TrialFunction, Identity, nabla_grad, grad, sym, det, div, dev, inv, tr, sqrt, conditional ,\
                 gt, dx, ds, inner, derivative, dot, ln, split, exp, eq, cos, acos, ge, le, FacetNormal, as_vector, variable)

# basix finite elements
import basix
from basix.ufl import element, mixed_element, quadrature_element




msh, markers, facet_tags = io.gmshio.read_from_msh("Cooksmembrane2d-nelem8-P2.msh", MPI.COMM_WORLD)

# coordinates of the nodes
x = ufl.SpatialCoordinate(msh)


# Define the volume integration measure "dx" 
# also specify the number of volume quadrature points.
dx = ufl.Measure('dx', domain=msh, metadata={'quadrature_degree': 4})
ds = ufl.Measure('ds', domain=msh, subdomain_data=facet_tags, metadata={'quadrature_degree': 4})



# FE Elements
# Quadratic element for displacement
###
# Define function spaces and elements
deg_u = 2
deg_p = deg_u-1

# displacement
U2 = element("Lagrange", msh.basix_cell(), deg_u, shape=(msh.geometry.dim,))
# pressure
P1 = element("Lagrange", msh.basix_cell(), deg_p)

# Mixed element
TH = mixed_element([U2, P1])
ME = functionspace(msh, TH)


V2 = functionspace(msh, U2) # Vector function space


# functions with DOFs at the current step
w = Function(ME)
u, p = split(w)

# functions with DOFs at the previous step
w_old = Function(ME)
u_old, p_old = split(w_old)

# functions with DOFs at the previous step
w_old2 = Function(ME)
u_old2, p_old2 = split(w_old2)

# Test functions
u_test, p_test = TestFunctions(ME)


# Trial functions
dw = TrialFunction(ME)


# Parameter values
# Parameter values
E = 240.565  # MPa
nu = 0.5 # Poisson's ratio
shearmod = E/(2*(1 + nu))
bulkmodInv = 3.0*(1.0-2*nu)/E

shearmod   = Constant(msh, PETSc.ScalarType(shearmod))
bulkmodInv = Constant(msh, PETSc.ScalarType(bulkmodInv))


t = 0.0
dt = 0.2
time_final = 1.0

num_steps = np.int32(time_final/dt) + 1


# Boundary conditions

'''
1 1 "leftedge"
1 2 "rightedge"
2 3 "solid"
'''

dofs_1_x = fem.locate_dofs_topological(ME.sub(0).sub(0), facet_tags.dim, facet_tags.find(1)) #ux
dofs_1_y = fem.locate_dofs_topological(ME.sub(0).sub(1), facet_tags.dim, facet_tags.find(1)) #uy

bc1_x = fem.dirichletbc(value=0.0, dofs=dofs_1_x, V=ME.sub(0).sub(0))
bc1_y = fem.dirichletbc(value=0.0, dofs=dofs_1_y, V=ME.sub(0).sub(1))

bcs = [bc1_x, bc1_y]


# traction
traction = Constant(msh, (0.0,0.0,0.0))


d = len(u)
I = Identity(d)
F = variable(I + grad(u))
J = det(F)

C = F.T*F

# Free Energy Function
Psi = shearmod/2*(J**(-2/3)*tr(C) - d) + p*(J-1-p*bulkmodInv/2)

#PK1 = diff(Psi, F) + p*J*inv(F.T)
PK1 = J**(-2/3)*shearmod*(F - 1/3*tr(C)*inv(F.T)) + J*p*inv(F.T)


# Weak form
Res  = inner(PK1, grad(u_test))*dx + inner((J-1 - p*bulkmodInv), p_test)*dx - dot(traction, u_test)*ds(2)
dRes = derivative(Res, w, dw)


# set up the nonlinear problem
problem = NonlinearProblem(Res, w, bcs, dRes)


# set the solver parameters
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-8
solver.atol = 1e-8
solver.max_it = 50
solver.report = True


#  The Krylov solver parameters.
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
opts[f"{option_prefix}ksp_max_it"] = 30
ksp.setFromOptions()




# displacement projection
U1 = element("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,))
#Vs = functionspace(msh, U1)

u_proj = Function(V2)
u_proj.name = "displacement"

p_proj = Function(functionspace(msh,P1))
p_proj.name = "pressure"


fname = "./results/Cooksmembrane2D-disp-.pvd"
VTKfile_Disp = io.VTKFile(msh.comm, fname, "w")
VTKfile_Disp.write_mesh(msh)

fname = "./results/Cooksmembrane2D-pres-.pvd"
VTKfile_Pres = io.VTKFile(msh.comm, fname, "w")
VTKfile_Pres.write_mesh(msh)

# function to write results to XDMF at time t
def writeResults_Disp(time_cur, timeStep):

    u_proj.interpolate(w.sub(0))

    #    file.close()
    VTKfile_Disp.write_function(u_proj)

# function to write results to XDMF at time t
def writeResults_Pres(time_cur, timeStep):
    
    p_proj.interpolate(w.sub(1))

    VTKfile_Pres.write_function(p_proj)



print("\n----------------------------\n")
print("Simulation has started")
print("\n----------------------------\n")

timeStep = 0
time_cur = 0.0

writeResults_Disp(time_cur, timeStep)
writeResults_Pres(time_cur, timeStep)


# loop over time steps
#for timeStep in range(num_steps):
while ( round(time_cur+dt, 6) <= time_final):
    # Update current time
    time_cur += dt
    timeStep += 1

    print("\n\n Load step = ", timeStep)
    print("     Time      = ", time_cur)

    traction.value = (0.0,6.25*time_cur,0.0)

    # Solve the problem
    (num_its, converged) = solver.solve(w)

    if converged:
        print(f"Converged in {num_its} iterations.")
    else:
        print(f"Not converged.")


    writeResults_Disp(time_cur, timeStep)
    writeResults_Pres(time_cur, timeStep)

    # save variables
    # velocity
    #w_old2.x.array[:] = w_old.x.array
    #w_old.x.array[:] = w.x.array

    w_old.vector.copy(w_old2.vector)
    w.vector.copy(w_old.vector)



VTKfile_Disp.close()
VTKfile_Pres.close()



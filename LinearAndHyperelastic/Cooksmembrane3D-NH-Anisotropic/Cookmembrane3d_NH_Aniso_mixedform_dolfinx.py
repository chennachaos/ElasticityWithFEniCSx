"""
@Problem: 2D Cook's membrane. A benchmark example.

@Formulation: Mixed displacement-pressure formulation.

@Material: Incompressible Linear elastic.

@author: Dr Chennakesava Kadapa

Created on Sun 18-Jun-2024
"""


# Import FEnicSx/dolfinx
import dolfinx

# For numerical arrays
import numpy as np

# For MPI-based parallelization
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# PETSc solvers
from petsc4py import PETSc

# specific functions from dolfinx modules
from dolfinx import fem, mesh, io, plot, log
from dolfinx.fem import (Constant, dirichletbc, Function, functionspace, Expression )
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile
#from dolfinx.io import VTXWriter

# specific functions from ufl modules
import ufl
from ufl import (TestFunctions, TrialFunction, Identity, grad, sym, det, div, dev, inv, tr, sqrt, conditional ,\
                 gt, dx, ds, inner, derivative, dot, ln, split, exp, eq, cos, acos, ge, le)

# basix finite elements
import basix
from basix.ufl import element, mixed_element, quadrature_element




msh, markers, facet_tags = io.gmshio.read_from_msh("Cooksmembrane3D-nelem4-P1.msh", MPI.COMM_WORLD)

# coordinates of the nodes
x = ufl.SpatialCoordinate(msh)


# Define the volume integration measure "dx" 
# also specify the number of volume quadrature points.
dx = ufl.Measure('dx', domain=msh, metadata={'quadrature_degree': 4})
ds = ufl.Measure('ds', domain=msh, subdomain_data=facet_tags, metadata={'quadrature_degree': 4})



# FE Elements
# Quadratic element for displacement
###
###
# Define function spaces and elements
deg_u = 3
deg_p = deg_u-1

# displacement
U2 = element("Lagrange", msh.basix_cell(), deg_u, shape=(3,))
# pressure
P1 = element("DG", msh.basix_cell(), 1)

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

# Test functions
u_test, p_test = TestFunctions(ME)


# Trial functions
dw = TrialFunction(ME)


# Boundary conditions

# Left edge is fixed

dofs_1_x = fem.locate_dofs_topological(ME.sub(0).sub(0), facet_tags.dim, facet_tags.find(1)) #ux
dofs_1_y = fem.locate_dofs_topological(ME.sub(0).sub(1), facet_tags.dim, facet_tags.find(1)) #uy
dofs_1_z = fem.locate_dofs_topological(ME.sub(0).sub(2), facet_tags.dim, facet_tags.find(1)) #uz

bc1_x = fem.dirichletbc(value=0.0, dofs=dofs_1_x, V=ME.sub(0).sub(0))
bc1_y = fem.dirichletbc(value=0.0, dofs=dofs_1_y, V=ME.sub(0).sub(1))
bc1_z = fem.dirichletbc(value=0.0, dofs=dofs_1_z, V=ME.sub(0).sub(2))

bcs = [bc1_x, bc1_y, bc1_z]


# Parameter values
E = 240.565  # MPa
nu = 0.4999 # Poisson's ratio
G = E/(2*(1 + nu))

eps = (3*(1 - 2*nu))/E 

mu = Constant(msh, G)
kappa_inv = Constant(msh, PETSc.ScalarType(eps))


mu = Constant(msh, PETSc.ScalarType(500.0))
lmbda = Constant(msh, PETSc.ScalarType(1000.0))

Cc = Constant(msh, PETSc.ScalarType(1000000.0))




t = 0.0
dt = 0.2
time_final = 1.0

# traction
traction = Constant(msh, (0.0,0.0,0.0))


# body force
f = Constant(msh, (0.0,0.0,0.0))


# dimension
d = len(u)


# Kinematics
#Id = Identity(d)
#F  = Id + grad(u)
#C  = F.T*F
#J  = det(F)

#Fbar = J**(-1.0/3)*F
#Cbar = Fbar.T*Fbar
#ICbar = tr(Cbar)


Id = ufl.variable(ufl.Identity(d))
F = ufl.variable(Id + ufl.grad(u))
C = ufl.variable(F.T * F)
Ic = ufl.variable(ufl.tr(C))
J  = ufl.variable(ufl.det(F))

Fbar = ufl.variable(J**(-1.0/3)*F)
Cbar = ufl.variable(Fbar.T*Fbar)
ICbar = ufl.variable(Cbar)


oneDsqrt3 = 1.0/sqrt(3.)

oneD3 = 1.0/3.0

#avec = [I[0,0],I[1,1],I[2,2]] /sqrt(3.0)

#avec = as_matrix([[oneDsqrt3,oneDsqrt3,oneDsqrt3],[oneDsqrt3,oneDsqrt3,oneDsqrt3],[oneDsqrt3,oneDsqrt3,oneDsqrt3]])

#M = outer(avec, avec)

M = ufl.as_tensor([[oneD3,oneD3,oneD3],[oneD3,oneD3,oneD3],[oneD3,oneD3,oneD3]])


CM = C*M

# Free Energy Function
Psi = mu/2*(tr(C) - 3 - 2*ln(J)) + 0.25*lmbda*(J*J-1-2*ln(J)) + p*(tr(CM)-1-p*p/2/Cc)


# First Piola-Kirchhoff stress
PK1 = ufl.diff(Psi, F)
#PK1 = J**(-2/3)*G*(F - 1/3*tr(C)*inv(F.T)) + J*p*inv(F.T)


# Weak form
Res = inner(PK1, grad(u_test))*dx + inner((tr(CM)-1 - p/Cc), p_test)*dx - dot(traction, u_test)*ds(2)

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


# Give fields descriptive names
u_v = w.sub(0)
u_v.name = "displacement"

p_v = w.sub(1)
p_v.name = "pressure"



# displacement projection
U1 = element("Lagrange", msh.basix_cell(), 1, shape=(3,))

Vs = functionspace(msh, U2)

u_proj = Function(V2)
u_proj.name = "displacement"

p_proj = Function(functionspace(msh, P1))
p_proj.name = "pressure"


fname = "Cooksmembrane3D-Aniso-disp-.pvd"
VTKfile_Disp = io.VTKFile(msh.comm, fname, "w")
VTKfile_Disp.write_mesh(msh)

fname = "Cooksmembrane3D-Aniso-pres-.pvd"
VTKfile_Pres = io.VTKFile(msh.comm, fname, "w")
VTKfile_Pres.write_mesh(msh)

# function to write results to XDMF at time t
def writeResults(time_cur, timeStep):

    # Displacement, pressure term
    u_proj.interpolate(w.sub(0))
    p_proj.interpolate(w.sub(1))

    #fname = "block3d-"+str(timeStep).zfill(6)+".vtu"
    #with io.VTKFile(msh.comm, fname, "w") as file:
    #    file.write_mesh(msh)
    #    file.write_function(u_proj)
    #    file.close()
    VTKfile_Disp.write_function(u_proj)
    VTKfile_Pres.write_function(p_proj)




print("\n----------------------------\n")
print("Simulation has started")
print("\n----------------------------\n")

timeStep = 0
time_cur = 0.0

writeResults(time_cur, timeStep)


# loop over time steps
#for timeStep in range(num_steps):
while ( round(time_cur+dt, 6) <= time_final):
    # Update current time
    time_cur += dt
    timeStep += 1

    print("\n\n Load step = ", timeStep)
    print("     Time      = ", time_cur)

    #traction.value = (0.0,0.0,-320.0*time_cur)
    traction.value = (0.0,250.0*time_cur,0.0)

    # Solve the problem
    # Compute solution
    (num_its, converged) = solver.solve(w)
    
    if converged:
        print(f"Converged in {num_its} iterations.")
    else:
        print(f"Not converged.")

    writeResults(time_cur, timeStep)

    # save variables
    # DOFs
    w_old.x.array[:] = w.x.array
    #velocity
    #v_temp.interpolate(v_expr)
    #v_old.x.array[:] = v_temp.x.array[:]
    #acceleration
    #a_temp.interpolate(a_expr)
    #a_old.x.array[:] = a_temp.x.array[:]


VTKfile_Disp.close()
VTKfile_Pres.close()

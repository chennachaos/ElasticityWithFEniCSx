"""
@Problem: 3D block under compression.

@Formulation: Mixed displacement-pressure formulation.

@Material: Incompressible Neo-Hookean model.

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




msh, markers, facet_tags = io.gmshio.read_from_msh("beam3d-magneto-soft-nelem20-Q2.msh", MPI.COMM_WORLD, 0, 3)

# coordinates of the nodes
x = ufl.SpatialCoordinate(msh)


# Define the volume integration measure "dx" 
# also specify the number of volume quadrature points.
dx = ufl.Measure('dx', domain=msh, subdomain_data=markers, metadata={'quadrature_degree': 4})
ds = ufl.Measure('ds', domain=msh, subdomain_data=facet_tags, metadata={'quadrature_degree': 4})



# FE Elements
# Quadratic element for displacement
###
###
# Define function spaces and elements
deg_u = 2
deg_p = deg_u-1

# displacement
U2 = element("Lagrange", msh.basix_cell(), deg_u, shape=(3,))
# pressure
P1 = element("Lagrange", msh.basix_cell(), deg_p)
P2 = element("Lagrange", msh.basix_cell(), deg_u)

# Mixed element
TH = mixed_element([U2, P1, P2])
ME = functionspace(msh, TH)


V2 = functionspace(msh, U2) # Vector function space

# functions with DOFs at the current step
w = Function(ME)
u, p, phi = split(w)

# functions with DOFs at the previous step
w_old = Function(ME)
u_old, p_old, phi_old = split(w_old)

w_old2 = Function(ME)
u_old2, p_old2, phi_old2 = split(w_old2)

# Test functions
u_test, p_test, phi_test = TestFunctions(ME)


# Trial functions
dw = TrialFunction(ME)

# Boundary conditions

'''
2 1 "fixed"
2 2 "fieldbc"
3 3 "activesolid"
3 4 "passivesolid"
'''

# Left edge is fixed
dofs_1_x = fem.locate_dofs_topological(ME.sub(0).sub(0), facet_tags.dim, facet_tags.find(1)) #uX
dofs_1_y = fem.locate_dofs_topological(ME.sub(0).sub(1), facet_tags.dim, facet_tags.find(1)) #uY
dofs_1_z = fem.locate_dofs_topological(ME.sub(0).sub(2), facet_tags.dim, facet_tags.find(1)) #uZ
dofs_1_phi = fem.locate_dofs_topological(ME.sub(2), facet_tags.dim, facet_tags.find(1)) #phi


dofs_2_phi = fem.locate_dofs_topological(ME.sub(2), facet_tags.dim, facet_tags.find(2)) #phi


bc1_x   = fem.dirichletbc(value=0.0, dofs=dofs_1_x, V=ME.sub(0).sub(0))
bc1_y   = fem.dirichletbc(value=0.0, dofs=dofs_1_y, V=ME.sub(0).sub(1))
bc1_z   = fem.dirichletbc(value=0.0, dofs=dofs_1_z, V=ME.sub(0).sub(2))
bc1_phi = fem.dirichletbc(value=0.0, dofs=dofs_1_phi, V=ME.sub(2))

phi_applied = Constant(msh, 0.0)

bc2_phi = fem.dirichletbc(value=phi_applied, dofs=dofs_2_phi, V=ME.sub(2))

bcs = [bc1_x, bc1_y, bc1_z, bc1_phi, bc2_phi]


# find facets for Neumann BCs
#facet_tags_4_indices = facet_tags.find(4)
#facet_tags_4_values  = np.hstack([np.full_like(facet_tags_4_indices, 4)])
#facet_tags_4_args    = np.argsort(facet_tags_4_indices)
#facet_tags_4 = mesh.meshtags(msh, msh.topology.dim-1, facet_tags_4_indices[facet_tags_4_args], facet_tags_4_values[facet_tags_4_args])

#ds4 = ufl.Measure('ds', domain=msh, subdomain_data=facet_tags_4)




# Parameter values
mu_active    = Constant(msh, 30000.0)
alpha_active = Constant(msh, -0.5)
beta_active  = Constant(msh, -4.0)
eta_active   = Constant(msh, -0.5)

mu_passive    = Constant(msh, 30000.0)
alpha_passive = Constant(msh, 0.0)
beta_passive  = Constant(msh, 0.0)
eta_passive   = Constant(msh, 0.0)

mu0 = Constant(msh, 1.2566)

kappa_inv = Constant(msh, 0.0)


t = 0.0
dt = 0.02
time_final = 1.0




# dimension
d = len(u)

# Kinematics
Id = ufl.variable(Identity(3))
F  = ufl.variable(Id + grad(u))
C  = ufl.variable(F.T*F)
J  = ufl.variable(det(F))

Cinv = ufl.variable(inv(C))
Fbar = ufl.variable(J**(-1.0/3)*F)
Cbar = ufl.variable(Fbar.T*Fbar)
CbarInv = ufl.variable(inv(Cbar))
ICbar = ufl.variable(tr(Cbar))

H = ufl.variable(-grad(phi))

#hh = ufl.variable(-grad(phi))
#H = ufl.variable(F.T*hh)

HHt = ufl.outer(H, H)

# Active layer
#
Psi_A  = 0.5*mu_active*(ICbar-3) + p*(J-1)
Psi_A += 0.5*mu0*J*inner(Cinv, HHt)
Psi_A += alpha_active*mu0*inner(Id, HHt) + beta_active*mu0*inner(Cbar, HHt) + eta_active*mu0*inner(CbarInv, HHt)

# First Piola-Kirchhoff stress
#PK1 = J**(-2/3)*mu*(F - 1/3*tr(C)*inv(F.T)) + J*p*inv(F.T)

PK1_A =  ufl.diff(Psi_A, F)
B_A   = -ufl.diff(Psi_A, H)

Res_A  = inner(PK1_A, grad(u_test))*dx(3) + inner(J-1-p*kappa_inv, p_test)*dx(3) - inner(B_A, -grad(phi_test))*dx(3)


# Passive layer
#
Psi_P  = 0.5*mu_passive*(ICbar-3) + p*(J-1)
Psi_P += 0.5*mu0*J*inner(inv(C), HHt)
Psi_P += alpha_passive*mu0*inner(Id, HHt) + beta_passive*mu0*inner(Cbar, HHt) + eta_passive*mu0*inner(CbarInv, HHt)

PK1_P =  ufl.diff(Psi_P, F)
B_P   = -ufl.diff(Psi_P, H)


Res_P = inner(PK1_P, grad(u_test))*dx(4) + inner(J-1-p*kappa_inv, p_test)*dx(4)- inner(B_P, -grad(phi_test))*dx(4)


# total Res
Res = Res_A + Res_P

dRes = derivative(Res, w, dw)


# set up the nonlinear problem
problem = NonlinearProblem(Res, w, bcs, dRes)
#problem = NonlinearProblem(Res, w, bcs)



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
#U1 = element("Lagrange", msh.basix_cell(), 1, shape=(3,))

u_proj = Function(functionspace(msh, U2))
u_proj.name = "displacement"

p_proj = Function(functionspace(msh, P1))
p_proj.name = "pressure"

phi_proj = Function(functionspace(msh, P2))
phi_proj.name = "phi"


fname = "./results/beam3d-disp-.pvd"
VTKfile_Disp = io.VTKFile(msh.comm, fname, "w")
#VTKfile_Disp.write_mesh(msh)

fname = "./results/beam3d-pres-.pvd"
VTKfile_Pres = io.VTKFile(msh.comm, fname, "w")
#VTKfile_Pres.write_mesh(msh)

fname = "./results/beam3d-phi-.pvd"
VTKfile_Phi = io.VTKFile(msh.comm, fname, "w")
#VTKfile_Phi.write_mesh(msh)


# function to write results to XDMF at time t
def writeResults(time_cur, timeStep):
    # Displacement
    u_proj.interpolate(w.sub(0))
    VTKfile_Disp.write_function(u_proj)
    #Pressure
    p_proj.interpolate(w.sub(1))
    VTKfile_Pres.write_function(p_proj)
    # Potential
    phi_proj.interpolate(w.sub(2))
    VTKfile_Phi.write_function(phi_proj)



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

    # solution predictor. This will improve convergence.
    if(timeStep > 2):
        w.x.array[:] = 2*w_old.x.array[:] - w_old2.x.array[:]

    phi_applied.value = 1500.0*time_cur

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
    w_old2.x.array[:] = w_old.x.array[:]
    w_old.x.array[:] = w.x.array
    #w.vector.copy(w_old.vector)



VTKfile_Disp.close()
VTKfile_Pres.close()
VTKfile_Phi.close()





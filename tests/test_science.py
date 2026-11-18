import cmath
import copy
import itertools
import math
import pytest
from molecular_materials_foundry.api import dispatch
from molecular_materials_foundry.errors import DomainError

def test_nmr_selected_channel_recovery(sample):
    result=dispatch('/nmr',sample('nmr'))
    assert result['signal'][0]==pytest.approx([2,0],abs=1e-12)

def test_nmr_rejects_other_channels_and_corrects_receivers():
    acquisitions=[]
    target=complex(2,-3)
    for i in range(4):
        phase=i*math.pi/2;receiver=(i+1)*math.pi/7
        value=(target*cmath.exp(1j*phase)+complex(4,5)*cmath.exp(-1j*phase))*cmath.exp(-1j*receiver)
        acquisitions.append({'phase_degrees':math.degrees(phase),'receiver_degrees':math.degrees(receiver),'signal':[[value.real,value.imag]]})
    result=dispatch('/nmr',{'channel':1,'acquisitions':acquisitions})
    assert result['signal'][0]==pytest.approx([2,-3])
    assert result['alias_modulus']==4

def test_nmr_incomplete_phase_plan_rejected(sample):
    p=sample('nmr');p['acquisitions'].pop()
    with pytest.raises(DomainError):dispatch('/nmr',p)

def test_isotope_distribution_matches_binomial():
    result=dispatch('/isotopes',{'composition':{'X':2},'populations':{'X':[[10,0.8],[11,0.2]]}})
    assert [x['coordinate'] for x in result['peaks']]==[20,21,22]
    assert [x['probability'] for x in result['peaks']]==pytest.approx([0.64,0.32,0.04])
    assert result['probability_sum']==pytest.approx(1)

def test_isotope_independent_atom_enumeration():
    populations={'X':[[10,0.5],[11,0.5]],'Y':[[1,0.25],[2,0.75]]}
    result=dispatch('/isotopes',{'composition':{'X':2,'Y':1},'populations':populations})
    expected={}
    for x1,x2,y in itertools.product(populations['X'],populations['X'],populations['Y']):
        mass=x1[0]+x2[0]+y[0];expected[mass]=expected.get(mass,0)+x1[1]*x2[1]*y[1]
    assert {p['coordinate']:p['probability'] for p in result['peaks']}==pytest.approx(expected)

def test_ion_charge_includes_electron_mass():
    result=dispatch('/isotopes',{'composition':{'C':1},'charge':2})
    assert result['peaks'][0]['coordinate']==pytest.approx((12-2*0.000548579909065)/2)

def test_isotope_rejects_invalid_population():
    with pytest.raises(DomainError):dispatch('/isotopes',{'composition':{'X':1},'populations':{'X':[[1,0.2],[2,0.2]]}})

def test_isotope_deterministic_population():
    result=dispatch('/isotopes',{'composition':{'X':200},'populations':{'X':[[1,1],[2,0]]}})
    assert result['peaks']==[{'coordinate':200.0,'probability':1.0}]

def test_stereochemistry_known_orientation_and_mirror(sample):
    p=sample('stereo');result=dispatch('/stereo',p)
    assert result['centers'][0]['descriptor']=='R'
    for atom in p['atoms']:atom['position'][0]*=-1
    assert dispatch('/stereo',p)['centers'][0]['descriptor']=='S'

def test_stereochemistry_numbering_and_rotation_invariant(sample):
    p=sample('stereo')
    for atom in p['atoms']:
        x,y,z=atom['position'];atom['position']=[-y,x,z]
    p['atoms'].reverse();p['bonds'].reverse()
    renaming={a['id']:str(i) for i,a in enumerate(p['atoms'])}
    for atom in p['atoms']:atom['id']=renaming[atom['id']]
    p['bonds']=[[renaming[a],renaming[b]] for a,b in p['bonds']];p['centers']=[renaming['c']]
    assert dispatch('/stereo',p)['centers'][0]['descriptor']=='R'

def test_stereo_tied_ligands_are_not_guessed(sample):
    p=sample('stereo');p['atoms'][2]['atomic_number']=53
    result=dispatch('/stereo',p)['centers'][0]
    assert result['descriptor'] is None and result['status']=='requires_deeper_ligand_comparison'

def test_stereo_planar_geometry_is_undetermined(sample):
    p=sample('stereo')
    for atom in p['atoms']:atom['position'][2]=0
    assert dispatch('/stereo',p)['centers'][0]['status']=='degenerate_geometry'

def test_reflection_weighted_mean_and_error(sample):
    result=dispatch('/reflections',sample('reflections'))['reflections']
    assert len(result)==1
    assert result[0]['intensity']==pytest.approx(12)
    assert result[0]['variance']==pytest.approx(2)
    assert result[0]['chi_squared']==pytest.approx(2)
    assert result[0]['observation_indices']==[0,1]

def test_reflections_preserve_friedel_pair_when_requested(sample):
    p=sample('reflections');p['friedel']=False
    assert len(dispatch('/reflections',p)['reflections'])==2

def test_reflections_require_closed_group(sample):
    p=sample('reflections');p['operations'].append([[0,-1,0],[1,0,0],[0,0,1]])
    with pytest.raises(DomainError):dispatch('/reflections',p)

def test_reflections_unequal_variances(sample):
    p=sample('reflections');p['observations'][1]['variance']=1
    result=dispatch('/reflections',p)['reflections'][0]
    assert result['intensity']==pytest.approx(13.2)
    assert result['variance']==pytest.approx(0.8)

def test_pure_water_is_neutral():
    r=dispatch('/speciation',{'acids':[]})
    assert r['pH']==pytest.approx(7)
    assert abs(r['charge_balance_residual'])<1e-18

def test_monoprotic_acid_matches_quadratic_limit():
    r=dispatch('/speciation',{'acids':[{'id':'a','total':0.01,'pka':4}]})
    k=1e-4;expected=2*k*0.01/(math.sqrt(k*k+4*k*0.01)+k)
    assert r['hydrogen']==pytest.approx(expected,rel=1e-7)
    assert r['species'][0]['neutral']+r['species'][0]['anion']==pytest.approx(0.01)
    assert abs(r['charge_balance_residual'])<1e-14

def test_strong_ions_and_multiple_acids_conserve_charge():
    r=dispatch('/speciation',{'acids':[{'id':'a','total':0.1,'pka':4},{'id':'b','total':0.2,'pka':7}],'strong_ion_charge':0.15})
    assert abs(r['charge_balance_residual'])<1e-12
    for species in r['species']:
        assert abs(species['mass_balance_residual'])<1e-14
        assert abs(species['equilibrium_residual'])<1e-14

def test_speciation_unsupported_model_rejected():
    with pytest.raises(DomainError):dispatch('/speciation',{'acids':[],'temperature_k':300})

def test_explicit_images_recover_boundary_crossing(sample):
    result=dispatch('/trajectories',sample('trajectories'))
    assert result['continuous_positions'][1][0]==pytest.approx([11,1,1])
    assert result['windows'][0]['mean_squared_displacement']==4

def test_bounded_inferred_images():
    p={'box':[10,10,10],'atoms':['a'],'frames':[{'positions':[[9,1,1]]},{'positions':[[1,1,1]]}],'max_step':3}
    r=dispatch('/trajectories',p)
    assert r['continuous_positions'][1][0]==(11,1,1)
    assert r['image_mode']=='bounded_step_assumption'

def test_trajectory_needs_information_for_images(sample):
    p=sample('trajectories')
    for frame in p['frames']:del frame['images']
    with pytest.raises(DomainError):dispatch('/trajectories',p)
    p['max_step']=5
    with pytest.raises(DomainError):dispatch('/trajectories',p)

def test_trajectory_detects_false_step_assumption():
    p={'box':[10,10,10],'atoms':['a'],'frames':[{'positions':[[9,1,1]]},{'positions':[[1,1,1]]}],'max_step':1}
    with pytest.raises(DomainError):dispatch('/trajectories',p)

def test_trajectory_molecular_bonds_validate_image_flags():
    p={'box':[10,10,10],'atoms':['a','b'],'frames':[{'positions':[[9.5,0,0],[0.5,0,0]],'images':[[0,0,0],[1,0,0]]}],
       'bonds':[['a','b']],'max_bond_length':1.1}
    dispatch('/trajectories',p)
    p['frames'][0]['images'][1]=[0,0,0]
    with pytest.raises(DomainError):dispatch('/trajectories',p)


def test_stereo_central_atom_outside_ligands_is_rejected(sample):
    p=sample('stereo');p['atoms'][0]['position']=[100,0,0]
    assert dispatch('/stereo',p)['centers'][0]['status']=='inconsistent_tetrahedral_center'


def test_atom_isotope_mass_changes_priority_without_input_order(sample):
    p=sample('stereo')
    p['atoms'][1].update(atomic_number=1,mass_number=3)
    p['atoms'][2].update(atomic_number=1,mass_number=2)
    p['atoms'][3].update(atomic_number=1,mass_number=1)
    p['atoms'][4].update(atomic_number=1,mass_number=4)
    result=dispatch('/stereo',p)['centers'][0]
    assert result['status']=='ok'
    assert result['priority_order']==['h','i','br','cl']


def test_reflections_nontrivial_swap_group():
    p={'operations':[[[1,0,0],[0,1,0],[0,0,1]],[[0,1,0],[1,0,0],[0,0,1]]],
       'observations':[{'hkl':[1,2,3],'intensity':8,'variance':1},
                       {'hkl':[2,1,3],'intensity':12,'variance':1}], 'friedel':False}
    result=dispatch('/reflections',p)['reflections']
    assert len(result)==1 and result[0]['hkl']==[1,2,3]
    assert result[0]['intensity']==10 and result[0]['variance']==0.5


def test_isotope_charge_cannot_produce_negative_mass():
    with pytest.raises(DomainError):
        dispatch('/isotopes',{'composition':{'X':1},'populations':{'X':[[1e-6,1]]},'charge':20})

from typing import Optional

from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np


@u.quantity_input(galPA=u.deg, galInc=u.deg, galDist=u.pc)
def deproject_coords(objectCoords, galCenter, galPA, galInc, galDist):
    """
    Function to deproject the coordinates of an object or 
    list of objects from the plane of the sky to the
    plane of the galaxy, given the coordinates of the 
    galaxy center, the position angle of the galaxy, 
    the inclination of the galaxy, and the distance to 
    the galaxy. The function returns the deprojected 
    (or galactocentric) radius and position angle of 
    the objects in the frame of the plane of the galaxy.

    Parameters:
    -----------
    objectCoords : SkyCoord or list_like
        The coordinates of the object(s) to be deprojected. 
        This can be a single SkyCoord object (with multiple coordinate pairs), 
        a list of SkyCoord objects, a list of u.Quantity objects, or a list 
        (or np.ndarray) of numbers with shape (2,N) or (N,2). 
        If the input is a (2,N) array, then each column is assumed 
        to be a coordinate pair. If the input is a (N, 2) array, 
        then each row is assumed to be a coordinate pair. 
        Any numbers inputted that are not a u.Quantity object are 
        assumed to be in degrees.

    galCenter : SkyCoord or list_like
        The coordinates of the galaxy center. This can be a single SkyCoord 
        object, a list of u.Quantity objects, a list of numbers with shape (2,), 
        or a u.Quantity object with shape (2,). If the input is a list of u.Quantity 
        objects or a list of numbers, then the first list is assumed to be the 
        RA and the second list is assumed to be the Dec. Any numbers inputted 
        that are not a u.Quantity object are assumed to be in degrees.

    galPA : u.Quantity (angle)
        The position angle of the galaxy. This is the angle between 
        the north direction and the major axis of the galaxy, measured 
        counterclockwise from the north direction.

    galInc : u.Quantity (angle)
        The inclination of the galaxy. This is the angle between the 
        plane of the galaxy and the line of sight.

    galDist : u.Quantity (distance)
        The distance to the galaxy.

    Returns:
    --------
    deproj_rad : u.Quantity (distance)
        The deprojected (or galactocentric) radius of the objects in the frame
        of the plane of the galaxy.
    deproj_PA : u.Quantity (angle)
        The deprojected (or galactocentric) position angle of the objects in the
        frame of the plane of the galaxy.
    """

    objectCoords = __convert_objectCoords(objectCoords)
    galCenter = __covert_galCenter(galCenter)
    galPA = galPA.to(u.deg)
    galInc = galInc.to(u.deg)
    galDist = galDist.to(u.pc)

    # Conversion factor from angular separation to physical separation in the plane of the sky
    phyCon = galDist.to(u.pc) / (206265*u.arcsec).to(u.deg)

    # Find the physical separation between the galaxy center and the objects' coordinates in the plane of the sky
    sky_sep = galCenter.separation(objectCoords) * phyCon
    # Find the position angle of the objects' coordinates with respect to the galaxy center in the plane of the sky
    sky_ang = galCenter.position_angle(objectCoords)

    # These are the x and y coordinates of the objects in the plane of the sky
    # treating the galaxy center as the origin, and the major axis of the galaxy as the 
    # x-axis, and the minor axis of the galaxy as the y-axis.
    sky_x = sky_sep * np.cos(sky_ang.to(u.radian))
    sky_y = sky_sep * np.sin(sky_ang.to(u.radian))

    # Now we apply the rotation matrix on the sky coordinates to get the coordinates into the frame
    # of the plane of the galaxy. The rotation is by the negative of the position angle of the galaxy 
    # because we want to rotate the coordinates in the opposite direction of the galaxy's rotation 
    # to get them into the frame of the galaxy.
    x_rot    =  sky_x * np.cos(-galPA.to(u.radian)) - sky_y * np.sin(-galPA.to(u.radian))
    y_rot    =  sky_x * np.sin(-galPA.to(u.radian)) + sky_y * np.cos(-galPA.to(u.radian))

    # Now we deproject from the oval projections of the galaxy on the plane of the sky to 
    # the assumed circular projection of the galaxy within its own frame.
    # The x coordinate is unchanged, but the y-coordinate is divided by the cosine of the 
    # inclination angle of the galaxy to account for the fact that the galaxy is tilted 
    # with respect to our line of sight.
    x_disk = x_rot
    y_disk = y_rot/np.cos(galInc.to(u.radian))

    # Now, we can find the deprojected (or galactocentric) radius and position angle of the objects
    # in the fram of the plane of the galaxy.
    deproj_rad = np.sqrt(x_disk**2 + y_disk**2)
    deproj_PA  = np.arctan2(y_disk, x_disk)

    return deproj_rad, deproj_PA.to(u.deg)


@u.quantity_input(objectRadii=u.pc, objectAngle=u.deg, galPA=u.deg, galInc=u.deg, galDist=u.pc)
def project_coords(objectRadii, objectAngle, galCenter, galPA, galInc, galDist):
    """
    Function to convert the galactocentric radius and position angle of an object or list of
    objects in the frame of the plane of the galaxy to RA and Dec coordinates in the plane of the sky, 
    given the coordinates of the galaxy center, the position angle of the galaxy, 
    the inclination of the galaxy, and the distance to the galaxy. The function returns 
    the RA and Dec coordinates of the objects in the plane of the sky.

    Parameters:
    -----------
    objectRadii : u.Quantity (distance)
        The galactocentric radius of the object(s) in the frame of the plane of the galaxy.
    objectAngle : u.Quantity (angle)
        The galactocentric position angle of the object(s) in the frame of the plane of the galaxy. 
        This is the angle between the north direction and the line connecting the galaxy center to
        the object, measured counterclockwise from the north direction.
    galCenter : SkyCoord or list_like
        The coordinates of the galaxy center. This can be a single SkyCoord object, a list of u.Quantity 
        objects, a list of numbers with shape (2,), or a u.Quantity object with shape (2,). If the 
        input is a list of u.Quantity objects or a list of numbers, then the first list is assumed to be 
        the RA and the second list is assumed to be the Dec. Any numbers inputted that are not a u.Quantity 
        object are assumed to be in degrees.
    galPA : u.Quantity (angle)
        The position angle of the galaxy. This is the angle between the north direction and the major
        axis of the galaxy, measured counterclockwise from the north direction.
    galInc : u.Quantity (angle)
        The inclination of the galaxy. This is the angle between the plane of the galaxy and the
        line of sight.
    galDist : u.Quantity (distance)
        The distance to the galaxy.

    Returns:
    --------
    ra : u.Quantity (angle)
        The RA coordinates of the objects in the plane of the sky.
    dec : u.Quantity (angle)
        The Dec coordinates of the objects in the plane of the sky.
    """

    objectRadii = objectRadii.to(u.pc)
    objectAngle = objectAngle.to(u.deg)
    galCenter = __covert_galCenter(galCenter)
    galPA = galPA.to(u.deg)
    galInc = galInc.to(u.deg)
    galDist = galDist.to(u.pc)

    # Conversion factor from angular separation to physical separation in the plane of the sky
    phyCon = galDist.to(u.pc) / (206265*u.arcsec).to(u.deg)

    # First we need to convert the galactocentric radius and position angle of the objects 
    # in the frame of the plane of the galaxy to x and y coordinates in the plane of the galaxy. 
    # The x coordinate is along the major axis of the galaxy, 
    # and the y coordinate is along the minor axis of the galaxy.
    # Y coordinate is multipled by the cosine of the inclination angle of the galaxy to account
    # for the fact that the galaxy is tilted with respect to our line of sight.
    x_disk = objectRadii * np.cos(objectAngle.to(u.radian))
    y_disk = objectRadii * np.sin(objectAngle.to(u.radian)) * np.cos(galInc.to(u.radian))

    # Now we apply the inverse of the rotation matrix on the disk coordinates to get the coordinates 
    # into the frame of the plane of the sky. The rotation is by the position angle of the galaxy 
    # to get them into the plane of the sky.
    x_rot = x_disk * np.cos(galPA.to(u.radian)) - y_disk * np.sin(galPA.to(u.radian))
    y_rot = x_disk * np.sin(galPA.to(u.radian)) + y_disk * np.cos(galPA.to(u.radian))

    # Now we can find the separation and position angle of the objects with respect to the galaxy center 
    # in the plane of the sky, and then convert those to RA and Dec coordinates.
    sky_r  = np.sqrt(x_rot**2 + y_rot**2)
    sky_PA = np.arctan2(y_rot, x_rot)
    coord = galCenter.directional_offset_by(sky_PA, sky_r/phyCon)

    return coord.ra, coord.dec


def __convert_objectCoords(objectCoords):
    # If the input is a list, then it can be a list of SkyCoord objects, u.Quantity objects, or a list of numbers with size (2,N) or (N,2).
    if isinstance(objectCoords, list):
        #If SkyCoord objects, then set coords to be a single SkyCoord object containing the same coordinates (i.e., list -> SkyCoord)
        if isinstance(objectCoords[0], SkyCoord):
            coords = SkyCoord([c.ra for c in objectCoords], [c.dec for c in objectCoords])

        #If the list is a list of lists, then it can be a list of u.Quantity objects or a list of numbers.
        elif isinstance(objectCoords[0], list):

            #If u.Quantity objects, then we need to convert to SkyCoord objects.
            if isinstance(objectCoords[0][0], u.Quantity):
                #If the list has only 2 lists, then we assume the first list is the RA and the second list is the Dec.
                if len(objectCoords) == 2:
                    coords = SkyCoord(objectCoords[0], objectCoords[1])
                #If the list has more than 2 lists, then we assume each list is a coordinate pair (i.e., [[RA1, Dec1], [RA2, Dec2], ...]).
                elif len(objectCoords[0]) == 2:
                    un = objectCoords[0][0].unit
                    ras = []
                    decs = []
                    for c in objectCoords:
                        ras.append(c[0].value)
                        decs.append(c[1].value)
                    coords = SkyCoord(np.array(ras)*un, np.array(decs)*un)
                else:
                    raise ValueError('Your object coordinates need to be in a list with shape (2, N) or (N, 2).')
            
            #If the list is a list of numbers, then we need to convert to SkyCoord objects.
            elif isinstance(objectCoords[0][0], (int, float)):
                #Luckily numpy can conver lists of numbers into a np.ndarray, so we do that
                objectCoords = np.array(objectCoords)
                #Then we can check the shape of the array to determine how to convert to SkyCoord objects.
                #If the first dimension has size 2, then we assume the first list is the RA and the second list is the Dec.
                if objectCoords.shape[0] == 2:
                    coords = SkyCoord(objectCoords[0]*u.deg, objectCoords[1]*u.deg)
                #If the second dimension has size 2, then we assume each list is a coordinate pair (i.e., [[RA1, Dec1], [RA2, Dec2], ...]).
                elif objectCoords.shape[1] == 2:
                    coords = SkyCoord(objectCoords[:,0]*u.deg, objectCoords[:,1]*u.deg)
                else:
                    raise ValueError('Your object coordinates need to be in a list of lists with shape (2, N) or (N, 2).')
            else:
                raise ValueError('Your object needs to be contain lists of u.Quantity objects of lists of numbers.')
        else:
            raise ValueError('Your object coordinates need to be in a list of SkyCoord objects or a list of lists with shape (2, N) or (N, 2).')
    
    #If the input is a np.ndarray, then it can be a list of numbers with size (2,N) or (N,2).
    elif isinstance(objectCoords, np.ndarray):
        #If the first dimension has size 2, then we assume the first list is the RA and the second list is the Dec.
        if objectCoords.shape[0] == 2:
            coords = SkyCoord(objectCoords[0]*u.deg, objectCoords[1]*u.deg)
        #If the second dimension has size 2, then we assume each list is a coordinate pair (i.e., [[RA1, Dec1], [RA2, Dec2], ...]).
        elif objectCoords.shape[1] == 2:
            coords = SkyCoord(objectCoords[:,0]*u.deg, objectCoords[:,1]*u.deg)
        else:
            raise ValueError('Your object coordinates need to be in a list of lists with shape (2, N) or (N, 2).')
    
    #If the input is a u.Quantity, then it needs to be a list of angles with shape (2,N) or (N,2).
    elif isinstance(objectCoords, u.Quantity):
        if u.get_physical_type(objectCoords.unit) == 'angle':
            #If the first dimension has size 2, then we assume the first list is the RA and the second list is the Dec.
            if objectCoords.shape[0] == 2:
                coords = SkyCoord(objectCoords[0], objectCoords[1])
            #If the second dimension has size 2, then we assume each list is a coordinate pair (i.e., [[RA1, Dec1], [RA2, Dec2], ...]).
            elif objectCoords.shape[1] == 2:
                coords = SkyCoord(objectCoords[:,0], objectCoords[:,1])
            else:
                raise ValueError('Your object coordinates need to be in a list of lists with shape (2, N) or (N, 2).')
        else:
            raise u.UnitsError('Your object coordinates need to be a list of angles.')

    #If the input is a SkyCoord object, then we can just use it as is.
    elif isinstance(objectCoords, SkyCoord):
        coords = objectCoords
    
    else:
        raise ValueError('Your object coordinates need to be: a list of SkyCoords/u.Quantity/numbers, a np.ndarray, a u.Quantity, or a SkyCoord object.')
    
    return coords

def __covert_galCenter(galCenter):
    #The galaxy center can be a SkyCoord object, a list of u.Quantity objects, a list of numbers, or a u.Quantity object.
    #We need to convert it to a SkyCoord object.

    #If the input is a SkyCoord object, then we can just use it as is.
    if isinstance(galCenter, SkyCoord):
        return galCenter
    
    #If the input is a list, then it can be a list of u.Quantity objects or a list of numbers.
    elif isinstance(galCenter, list):
        #If the list is a list of u.Quantity objects, then we can convert it to a SkyCoord object.
        if isinstance(galCenter[0], u.Quantity):
            return SkyCoord(galCenter[0], galCenter[1])
        #If the list is a list of numbers, then we can convert it to a SkyCoord object by assuming the numbers are in degrees.
        elif isinstance(galCenter[0], (int, float)):
            return SkyCoord(galCenter[0]*u.deg, galCenter[1]*u.deg)
        else:
            raise ValueError('Your galaxy center needs to be a list of u.Quantity objects or a list of numbers.')
        
    #If the input is a u.Quantity, then it needs to be a list of angles with shape (2,).
    elif isinstance(galCenter, u.Quantity):
        if u.get_physical_type(galCenter.unit) == 'angle':
            return SkyCoord(galCenter[0], galCenter[1])
        else:
            raise u.UnitsError('Your galaxy center needs to be a list of angles.')
    else:
        raise ValueError('Your galaxy center needs to be a SkyCoord object, a list of u.Quantity objects or numbers, or a u.Quantity object.')
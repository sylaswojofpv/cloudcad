

<img width="960" height="495" alt="CloudCAD" src="https://github.com/user-attachments/assets/38f4cf6d-a291-4d76-bf99-d8ca16daf5a5" />


POINT CLOUD MODELER
===================

A desktop 3D modeling program for building clean, editable models on top of a
scanned point cloud. Load one or more point clouds (.xyz and similar), then use
the cloud as a reference: click points on it to create faces and best-fit
solids, lasso part of it to build a mesh, snap geometry to its surface, measure
it, mirror bodies onto it, and combine everything with boolean operations.

Written in Python (PySide6 + OpenGL). Windows is the primary platform.


TABLE OF CONTENTS
-----------------
   1. Features at a glance
   2. Installation
   3. Starting the program
   4. Quick start
   5. The window
   6. Moving around the 3D view
   7. Loading point clouds
   8. Tools
        8.1  Select              (Q)
        8.2  Move / Transform    (W)
        8.3  Face from points    (P)
        8.4  Best-fit solid      (B)
        8.5  Mesh from points    (G)   lasso + Poisson / Delaunay
        8.6  Measure             (R)
        8.7  Constraints         (K)
        8.8  Mirror body         (T)
   9. Editing operations (subdivide, extrude, delete, ...)
  10. Combining bodies: booleans and merge / stitch
  11. The Bodies list
  12. Undo and redo
  13. Saving, opening and exporting
  14. Keyboard shortcuts
  15. Project files
  16. Tips, limits and troubleshooting


1. FEATURES AT A GLANCE
-----------------------
Point clouds
  - Load many files at once, show / hide each one, drag and drop files in.
  - Reads x y z in the first three columns; extra columns (normals, colors,
    intensity) are ignored. Header lines, comments, blank lines and space /
    tab / comma / semicolon separators are handled.
  - Handles very large clouds: only part is drawn for speed, but clicks pick
    from every point.

Modeling on the cloud
  - Face from points: click points on the cloud, get a face.
  - Best-fit solids: box, cylinder, cone, sphere, plane, loft, extrusion and
    revolve, each fitted to the points you click (least-squares, so the points
    do not need to be perfect).
  - Mesh from points: lasso or polygon-select part of the cloud and build a
    triangle mesh with Poisson surface reconstruction or Delaunay triangulation.
    Detail and triangle count are adjustable; normals are estimated and
    oriented automatically.
  - Move vertices, edges and faces with handles that align to the selection.
    Optionally the geometry follows the cloud surface while you drag, or snaps
    to it.

Freeform editing
  - Vertex / edge / face selection, subdivide, extrude (faces and edges), fill,
    weld, flip normals, and a delete that removes the element without tearing
    holes (faces merge, edges join).

Combining
  - Boolean union, difference and intersect on closed solids, with results kept
    in folders so every boolean can be undone later.
  - Merge / stitch: join bodies by welding and stitching their vertices, edges
    and faces without re-meshing.

Measuring and constraining
  - Distance, path length, angle, circle and sphere diameter (fitted), edge
    length, angle between faces, and size / area / volume of a shape.
  - Constraints: parallel, perpendicular, collinear / coplanar, distance, angle,
    equal length, midpoint, align to axis, fixed. They are re-solved whenever
    you edit.

Mirroring
  - Mirror a body through an adjustable plane, auto-align the plane to the
    point cloud, optionally keep the mirror linked so it follows later edits.

Organizing
  - A Bodies list showing every point cloud and body with visibility tick boxes,
    folders for boolean results, renaming, and one-click undo of a boolean or
    merge.
  - Save the whole model as a .json file; export to Wavefront OBJ.


2. INSTALLATION
---------------
You need Python 3.9 or newer. Python 3.10 - 3.12 is the safest choice: some
packages (Open3D in particular) publish wheels for brand-new Python versions
later than for older ones.

Install everything with:

    pip install -r requirements.txt

or one by one:

    pip install PySide6 PyOpenGL numpy scipy manifold3d open3d shapely

What each package is for:

    PySide6      the user interface (required)
    PyOpenGL     drawing in the 3D view (required)
    numpy        numerical work (required)
    scipy        fitting, nearest-neighbor search, Delaunay (required)
    manifold3d   boolean union / difference / intersect, and the union step of
                 Merge (only needed for those)
    open3d       Poisson surface reconstruction in "Mesh from points"
                 (only needed for the Poisson method; Delaunay does not need it)
    shapely      merging bodies whose faces partly overlap (optional; Merge
                 tells you if it would have helped)

A feature that needs a package you have not installed shows a message naming
the package; everything else keeps working.

Graphics: the program asks for an OpenGL 2.1 (compatibility) context with
depth buffer and 4x multisampling. Any ordinary graphics card and up-to-date
driver will do. If your driver does not support multisampling and the window
stays blank, update the graphics driver.


3. STARTING THE PROGRAM
-----------------------
From a command prompt in the program folder:

    python modeler.py

You can pass point-cloud files on the command line; they are all loaded:

    python modeler.py scan_north.xyz scan_south.xyz

Run it with "python modeler.py" (not by double-clicking) the first time, so
that any error message stays visible in the console.


4. QUICK START
--------------
1. File > Open point clouds (Ctrl+O), pick one or more files. The camera frames
   the cloud and a starting plane appears.
2. Put a face on the cloud: press P, click 4 points on the cloud, a face is
   made (Enter finishes early if you set it to manual).
   Or make a solid: press B, pick a shape (box, cylinder, ...), click several
   points on the object's surface and press Enter. The shape is fitted to your
   points.
   Or build a mesh from a region: press G, click a polygon around part of the
   cloud, right-click to close it, then press Enter.
3. Move things: press W, click a face, vertex or edge and drag it, or drag one
   of the colored handles to move along one axis.
4. Shape it: Subdivide (S), Extrude (E), Delete (Del) and so on; see section 9.
5. Combine: select two solids and use Union / Difference / Intersect (section 10).
6. File > Save model (Ctrl+S) to keep your work, File > Export OBJ to use the
   model elsewhere.


5. THE WINDOW
-------------
 - Center: the 3D view. A small hint line in its corner tells you what the
   current tool does (turn it on / off with H).
 - Toolbar: the tools and the selection modes (Vertex / Edge / Face).
 - Left panel (F2): the Bodies list - point clouds, bodies and boolean folders.
 - Right panel (F3): Properties. It shows only the options for the tool that
   is active. Options that most people rarely need are folded away (click the
   header to open them).
 - F4 hides or shows both panels at once, which gives the 3D view the whole
   window.
 - The status bar at the bottom shows messages: what was just done, and why
   something could not be done.

The Properties panel scrolls if the window is short.


6. MOVING AROUND THE 3D VIEW
----------------------------
    Orbit                 middle mouse button drag, or Alt + left drag
    Pan                   Shift + middle drag, or right mouse drag
    Zoom                  mouse wheel (zooms toward the cursor)
    Frame everything      Home
    Standard views        Ctrl+1 top, Ctrl+2 front, Ctrl+3 right

The view is Z-up. The grid sits at the lowest point of the loaded cloud.


7. LOADING POINT CLOUDS
-----------------------
File > Open point clouds (Ctrl+O) replaces the loaded clouds with the files you
choose (you can pick several at once). File > Add point clouds (Ctrl+Shift+O)
keeps what is loaded and adds more. You can also drag files from a folder onto
the window - they are added to the scene.

Accepted: .xyz, .txt, .csv, .pts, or any text file with numbers.
  - x y z are read from the first three columns.
  - Anything after that (nx ny nz, r g b, intensity, ...) is ignored, so files
    with or without normals both load.
  - Header lines, # and // comments and blank lines are skipped.
  - Columns may be separated by spaces, tabs, commas or semicolons.
  - Lines that cannot be read are skipped; files with no readable points are
    reported and the others still load.

Coordinates: the model is stored relative to the center of the first cloud(s)
loaded, so very large coordinates (map / survey coordinates) do not cause
jitter. Everything stays consistent: if you open a model and a cloud in either
order they line up. "World coordinates" (a checkbox in the Selection panel)
shows positions in the cloud's original coordinates, and OBJ export can write
them back (section 13).

Big clouds: up to about 3 million points are drawn (evenly thinned if there are
more) and up to about 400,000 take part while the mouse hovers or drags. Clicks
pick from all the points.

Showing and hiding: every cloud has a tick box in the Bodies list. Hidden
clouds are not drawn, cannot be picked and are ignored by surface following.

Display options (Properties panel > Display, folded by default):
    Point size, Pick radius (how close in pixels a click must be to pick a
    cloud point), Color points by height, and Opacity of the model's faces
    (lower it to see the cloud through your surfaces).


8. TOOLS
--------
Pick a tool from the toolbar or press its key. Esc cancels what you are doing.
Shift or Ctrl clicks usually add to or remove from a selection.

Selection modes (used by Select and Move, and by several others):
    1 = Vertex,  2 = Edge,  3 = Face
When you switch mode, the selection is converted where possible.


8.1  SELECT  (Q)
----------------
Click a vertex, edge or face (depending on the mode) to select it.
Shift+click adds or removes. Click empty space to clear. Selected faces are
orange.

The Selection panel shows how many things are selected and the X / Y / Z
position of the selection (the center when several vertices are selected).
You can type new values into those fields to move the selection exactly.

Select a face of a body and the whole body can be picked from the Bodies list;
see section 11.


8.2  MOVE / TRANSFORM  (W)
--------------------------
Select vertices, edges or faces and move them:

  - Drag the colored arrows (handles) to move along one axis.
  - Drag the element itself to move freely in the plane of the screen.
  - While dragging, press X, Y or Z to lock the move to that axis; press the
    same key again to unlock.
  - Hold Ctrl while dragging to snap to the nearest point of the cloud.
  - Esc while dragging cancels the move.

Aligned handles (checkbox "Aligned handles", on by default): the handles line
up with what is selected. For faces, edges and vertices the blue Z handle is the
surface normal and X follows an edge. For a whole closed solid the handles
follow the solid's own principal axes (a tilted cylinder gets a handle along its
axis). With the box off you get the world X / Y / Z. The X / Y / Z lock keys
use the same axes as the handles.

Follow surface (C): tick "Follow surface" (or press C) and what you drag slides
along the point cloud's surface instead of moving freely, so a face "travels"
over the scan. Each vertex is projected onto a plane fitted to its nearest
cloud points. Options:
    Offset      height above the surface (so a face can hover above the scan)
    Smoothing   how many nearby cloud points are averaged (more = smoother)
Following applies to free dragging. Dragging a handle, or locking an axis with
X / Y / Z, still moves exactly along that axis, so you can adjust height with
following switched on. Press C during a drag to switch it on and off.

Snap to surface (Shift+S) moves the selection onto the cloud surface in one go,
without dragging.


8.3  FACE FROM POINTS  (P)
--------------------------
Click points on the cloud (or existing mesh vertices) to make a face.
  - After the number of clicks set in "Finish after" (default 4) the face is
    made. Set it to "manual" and press Enter to finish with any number of
    points (3 or more).
  - Ctrl+click ignores mesh vertices and picks only cloud points (useful when a
    mesh vertex is in the way).
  - Clicking existing mesh vertices reuses them, so neighboring faces connect.
  - Backspace removes the last point, Esc cancels.
  - Points that would make a "bow tie" are reordered automatically; the face is
    turned to face the camera.


8.4  BEST-FIT SOLID  (B)
------------------------
Choose a shape, click points on the surface of the real object, press Enter.
The shape is fitted to the points with a least-squares fit, so the points do
not need to be perfectly aligned. A preview of the fitted shape appears as soon
as you have enough points and updates with every click; the status line shows
the fitted size and the error.

Shapes and the least number of points:
    Box (3)         the smallest oriented box around the points. Click the
                    visible corners and edge ends of the real box.
    Cylinder (5)    axis, radius and length. Points can be spread over a
                    partial arc.
    Cone (6)        apex, axis, angle and height. 8 or more points recommended.
    Sphere (4)      center and radius. Spread the points around it.
    Plane (3)       a flat sheet fitted to points on a flat surface.
    Loft (8)        a surface through sections fitted over points spread all
                    around and along a shape.
    Extrusion (6)   a 2D profile, fitted to points all around the shape's sides,
                    pushed along a direction.
    Revolve (8)     a profile curve spun around a fitted axis: vases, bottles,
                    bowls, domes. Front-side points are enough.

Options (Properties panel):
    Segments        how smooth round shapes are (8 - 96)
    Type            Linear or Fitted (curved) - for loft and extrusion
    Along length    segments along a loft / extrusion (auto by default)

Bodies made this way remember the points you clicked. Right-click the body in
the Bodies list and choose "Edit points..." to adjust those points; the shape
is refitted.

After a solid is created the Move tool is active with the whole solid selected,
and everything in section 9 works on its faces and vertices.


8.5  MESH FROM POINTS  (G)
--------------------------
Builds a triangle mesh from a region of the point cloud.

  1. Select the points. Two ways (choose in the panel):
       Polygon   click the corners of a polygon around the cloud points you
                 want, then right-click to close it.
       Freehand  hold the left mouse button and draw a lasso.
     Shift + draw adds to the selection, Ctrl + draw removes from it. The
     selected points are highlighted and counted.
  2. Choose the method in the panel:
       Poisson             a smooth, watertight-style surface reconstruction
                           (needs Open3D). Best for objects and general scans.
       Delaunay (best fit) triangulates the points on their best-fit plane.
                           Good for walls and roughly flat patches.
       Delaunay (XY)       triangulates in the world XY plane. Good for
                           terrain-like clouds seen from above.
  3. Adjust detail (see below), then press Enter or "Add mesh to model".
     Esc clears the selection.

Detail and triangle count:
    Detail          octree depth for Poisson - low = coarse, smooth, fewer
                    triangles; high = fine, many triangles.
    Max triangles   upper limit (0 = unlimited). If the mesh has more, it is
                    simplified down to this number.
    Max edge length (Delaunay) drops long thin triangles that bridge gaps.

Normals: normal directions are calculated for you. A plane is fitted to each
point's neighbors, the signs are made consistent across the whole surface, and
then oriented: a surface that wraps around a center (an object) points outward;
a one-sided scan (terrain, a wall) points toward your view. You can force
Outward or Inward in the panel. The triangles are wound so that the mesh's face
normals follow the point normals.

Trim: with "Trim" on, the low-density skin Poisson invents where there are no
points is cut away, so an open patch does not turn into a closed blob.

The new mesh is added as its own body ("Poisson mesh N" or "Mesh N").


8.6  MEASURE  (R)
-----------------
Pick the kind of measurement, click points, read the numbers next to the lines
drawn in the 3D view. Measurements stay in the scene and are saved with the
model.

    Distance (2 points)        length, delta X / Y / Z, horizontal distance
    Path length (many points)  total length of a polyline; press Enter to finish
    Angle (3 points)           the middle point is the corner
    Diameter / radius (circle) 3 or more points on a rim, press Enter - a circle
                               is fitted to all of them and its center becomes
                               a point other measurements can snap to
    Diameter (sphere)          4 or more points around a sphere, press Enter -
                               fitted the same way
    Edge length                click a mesh edge
    Angle between faces        click two faces

Clicks snap to cloud points, shape vertices, points on edges (midpoints snap),
points on faces, and to the centers of measured circles and spheres.
Ctrl = cloud points only.

"Diameter of selected shape" fits a sphere or cylinder to the points of the
selected shape and gives its diameter. For a selection of faces, the panel also
shows the size, area and (for a closed shape) volume.

Measurement display: units label, a scale factor (to show millimeters, feet,
...) and the number of decimals are set in the panel and saved with the model.


8.7  CONSTRAINTS  (K)
---------------------
Constraints make geometry obey a relationship, and keep obeying it while you
keep editing.

    Parallel              two edges / faces end up parallel
    Perpendicular         two edges / faces at 90 degrees
    Collinear / Coplanar  edges on one line, faces in one plane, a vertex on a
                          line or plane
    Distance              between two vertices / edges / faces, or the length of
                          one edge
    Angle                 between two edges / faces (or an edge and a face)
    Equal length          two edges the same length
    Midpoint              a vertex at the middle of an edge
    Align to axis         lay an edge along, or turn a face toward, a world axis
    Fixed                 lock vertices / edges / faces in place

How: choose the kind, then click vertices, edges and faces in order. The FIRST
object you pick stays where it is; the others move to satisfy the constraint.
Press Enter to apply (Backspace undoes the last pick). Distance and angle take a
value you type; "Align" takes an axis.

The panel lists every constraint. You can select one, switch it off and on,
change its value, or delete it. A constraint that cannot be satisfied is marked
with [!] and the reason. If the geometry a constraint used is deleted, the
constraint is removed and you are told. Constraints are saved with the model.


8.8  MIRROR BODY  (T)
---------------------
Creates a mirrored copy of one or more bodies, e.g. the other half of a
symmetrical object scanned from one side.

  1. Select the bodies (Ctrl+click bodies to add or remove them).
  2. A ghost of the mirrored body appears. Drag to slide the mirror plane;
     Shift + drag tilts it. The panel also has fields for the plane's position
     and tilt.
  3. Optionally press "Auto-align to the point cloud": the plane is adjusted so
     the mirrored copy lands on the scan.
  4. Press Enter ("Create mirrored body").

Options:
    Link        the mirrored body keeps following the original, so later edits
                to the original also appear in the mirror (the links are listed
                in the panel and can be cleared)
    Weld        join the two halves along the mirror plane
    Keep        keep the original body (on by default)

Mirror links are saved with the model.


9. EDITING OPERATIONS
---------------------
Operations work on the current selection. The buttons in the Selection panel
only show what makes sense for the selection mode.

    Subdivide (S)     face mode. Splits faces into quads (a center vertex plus
                      edge midpoints). Neighboring faces get the midpoint added
                      so the surface stays closed.
    Extrude (E)       face or edge mode. Faces: the face is copied and side
                      walls are added; drag the blue Z handle to pull it out.
                      Edges: a new quad is pulled out of the edge.
    Fill face (F)     vertex mode. Makes a face from the selected vertices.
    Weld (M)          vertex mode. Merges the selected vertices into one.
    Flip (N)          face mode. Reverses the face normals.
    Snap (Shift+S)    snaps the selection onto the cloud surface.
    Delete (Del)      see below.
    Plane (Shift+A)   Create > Plane adds a new flat plane at the view center.
    Remove unused vertices   (Model menu) cleans up leftover vertices from
                      deleted shapes.

Delete removes the element without tearing the surface:
    Edge mode     the two faces next to the edge merge into one. An end vertex
                  left with only two roughly straight edges is removed so those
                  edges become one. Sharp corners stay. Deleting a border edge
                  (only one face next to it) removes that face, and the status
                  line tells you.
    Vertex mode   the faces around the vertex merge into one polygon; a vertex
                  with just two edges is removed and its edges join. If removing
                  a vertex would tear the surface it is kept and you are told.
    Face mode     the face is deleted (and vertices only that face used).
This is the way to remove unneeded edges after a Subdivide: select the edges in
Edge mode and press Delete.


10. COMBINING BODIES
--------------------
Select the bodies first: click a face of one body, then Shift+click a face of
each other one in the view - or click bodies in the Bodies list and Ctrl/Shift+
click more.

BOOLEANS (Properties > Combine solids, or the Model menu)
    Union        joins the solids into one
    Difference   the first body you picked, minus all the others (the status
                 line says which one is cut)
    Intersect    keeps only the volume the solids share

  - Works on closed solids. Faces left loose on a solid (a plane touching it)
    are ignored; if a body is not closed you get a message such as "Solid 1
    can't be used: it is not closed (4 open edges)" and nothing changes. A
    solid that is inside-out is flipped automatically.
  - Where solids cross, the new edges are computed for you. Flat triangles are
    merged back into quads and larger polygons and needless straight vertices
    are removed, so the result stays easy to edit. Curved surfaces stay
    triangulated.
  - The result goes into its own folder in the Bodies list together with the
    original solids (greyed). The folder's Undo button removes the result and
    brings the originals back exactly. You can do booleans on results too
    (folders nest); undoing an inner one first undoes the later ones that use it.
  - Needs: pip install manifold3d

MERGE / STITCH (Ctrl+J, or "Merge / stitch" in the panel)
    Joins the selected bodies into ONE body without re-meshing: vertices that
    touch are welded, vertices are set into the other body's edges, faces that
    touch are removed. The rest of each body keeps its own faces. If the
    bodies overlap, or touching faces do not line up well enough, the
    overlapping part is fused with a union automatically.
    Options:
        Merge tolerance    how close counts as touching (auto = 0.2% of size)
        Bridge gaps up to  joins open rims that face each other across a gap
                           (e.g. two tubes without end caps) with a strip of
                           faces (off by default)
    Right-click the merged body in the Bodies list and choose "Undo merge" to
    get the original bodies back.


11. THE BODIES LIST
-------------------
The left panel (F2) lists everything in the scene.

Point clouds    one row per loaded cloud with its point count and a tick box
                for visibility.
Bodies          one row per body (a plane, a face, a fitted solid, a mesh...).
                The tick box shows or hides the body. A hidden body is removed
                from the view and from picking, and is not exported.
Boolean folder  one folder per boolean: the result (tick box) and the original
                solids it was made from (greyed). Press Undo on the folder to
                give the originals back. Folders inside other folders can be
                undone once the outer one is undone.

Things you can do:
    Click a body            selects it in the 3D view (Ctrl or Shift + click adds
                            or removes more bodies)
    Click a folder          selects its result
    Double-click or F2      rename a body, folder or point cloud
    Right-click             Rename, Select, Undo merge, Edit points (for bodies
                            made with the Best-fit solid tool), Undo (folders)

Use the list to pick bodies for a boolean or merge, which is easier than
clicking in a crowded view.


12. UNDO AND REDO
-----------------
    Ctrl+Z   undo          Ctrl+Y   redo

Ctrl+Z works in this order:
  1. If you are picking points (face from points, a solid, measurements,
     constraints, ...), it removes the last picked point.
  2. Otherwise, if you just selected something by clicking, it un-selects the
     last clicked element (newest first, in vertex, edge and face modes).
  3. Otherwise it undoes the last edit.
Up to 100 steps are kept. Showing or hiding bodies and the results of booleans,
merges and mirrors are all part of the history.


13. SAVING, OPENING, EXPORTING
------------------------------
File > Save model (Ctrl+S) writes a .json file containing:
    - all geometry
    - the Bodies list: names, visibility, boolean folders and the original
      solids inside them, merge history, and the points behind fitted solids
    - measurements and the measurement units / scale / decimals
    - constraints
    - mirror links

File > Open model reads it back. Point clouds are NOT stored in the model file:
load the cloud files again (before or after opening the model - they line up).

File > Export OBJ writes a Wavefront .obj with the visible geometry:
    Original coordinates   the model in the point cloud's own coordinates, so
                           it lines up with the scan in other software
    Cloud-centered         relative to the center of the cloud (small numbers)
Hidden bodies, and the original solids stored inside boolean folders, are not
exported.


14. KEYBOARD SHORTCUTS
----------------------
Tools
    Q  Select                      W  Move / Transform
    P  Face from points            B  Best-fit solid
    G  Mesh from points            R  Measure
    K  Constraints                 T  Mirror body
    1 / 2 / 3   Vertex / Edge / Face mode

Editing
    S            Subdivide faces
    E            Extrude faces / edges
    F            Fill face from selected vertices
    M            Weld selected vertices
    N            Flip normals
    Shift+S      Snap selection to the cloud surface
    Shift+A      Add a plane
    Ctrl+J       Merge selected bodies (stitch)
    Delete       Delete selection
    C            Follow cloud surface while moving (on / off)

While dragging with Move
    X / Y / Z    lock to an axis (again = unlock)
    Ctrl         snap to the nearest cloud point
    Esc          cancel the move

While picking points or building
    Enter        finish (create the face / solid / mesh / measurement /
                 constraint / mirror)
    Backspace    remove the last picked point
    Esc          cancel / clear

View
    Home         frame everything
    Ctrl+1/2/3   top / front / right view
    H            show / hide the hint overlay
    F2           show / hide the Bodies list
    F3           show / hide the Properties panel
    F4           show / hide both panels

File and history
    Ctrl+O          open point clouds       Ctrl+Shift+O   add point clouds
    Ctrl+S          save model              Ctrl+Q         quit
    Ctrl+Z          undo                    Ctrl+Y         redo

Mouse
    Middle drag = orbit       Shift + middle drag or right drag = pan
    Alt + left drag = orbit   Wheel = zoom to cursor


15. PROJECT FILES
-----------------
    modeler.py        the application: main window, 3D view, tools, panels
    mesh_core.py      point cloud loader, the mesh (faces, vertices, edit
                      operations), camera and picking math
    solids.py         best-fit shapes (box, cylinder, cone, sphere, plane,
                      loft, extrusion, revolve) and their meshes
    booleans.py       union / difference / intersect, triangle merging
    stitch.py         merge / stitch of bodies without re-meshing
    reconstruct.py    Poisson and Delaunay mesh from selected points,
                      point normals
    measure.py        measurements
    constraints.py    geometric constraints and their solver
    mirror.py         mirror plane, fitting the plane to the cloud
    bodies.py         the data behind the Bodies list (bodies, folders, merges)
    bodies_panel.py   the Bodies list panel
    requirements.txt  Python packages


16. TIPS, LIMITS AND TROUBLESHOOTING
------------------------------------
Picking points
  - If you cannot reach a particular cloud point, zoom in and click right on
    it, or raise Display > Pick radius. Hold Ctrl to ignore mesh vertices that
    are in the way.
  - Many solid shapes need points spread around the object, not all on one
    face. A box fits best if you click its visible corners.
  - Press Ctrl+Z to take back a wrong click.

Booleans and merge
  - Booleans need CLOSED solids. Solids created with the Best-fit solid tool
    are closed; a mesh with open edges or a loose fin is not.
  - A boolean on a face with a hole leaves triangles around the hole, and
    curved surfaces stay triangulated. Faces here cannot have inner holes.
  - Merge is the choice when you want to keep each body's own faces; Union
    re-computes the surface where the bodies cross.

Performance
  - Meshes with tens of thousands of faces are still editable but dragging
    gets slower as the count grows. Use a lower Detail or Max triangles for
    "Mesh from points" if the view becomes sluggish.
  - Hide point clouds you are not using (tick boxes in the Bodies list).

Common messages
  - "... needs Open3D / manifold3d / shapely": install the named package
    (pip install open3d, pip install manifold3d, pip install shapely).
  - "ModuleNotFoundError: No module named 'mesh_core'" (or another program
    file): keep all the .py files from this project together in one folder and
    start the program from that folder.
  - A blank 3D view: update your graphics driver; make sure no error appeared
    in the console you started the program from.

Not stored in the model file: the point clouds themselves, the camera position,
and display settings.

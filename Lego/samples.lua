--[[

 LDCad official samples script.
 Copyright Roland Melkert 2016-2019
 Free for non-commercial use.
 Version 2019-05-25

 This script registers a number of example orientated macro and animation scripts.

 Currently it holds the following macro's:
 - Hello world
     This macro will display a message dialog containing the current LDCad and lua version.

 - Selection circle
     This macro will ask for a radius and then use that radius to reposition the items in the current selection into a circle.

 - Add wall
     This macro will add a wall of 2x4 bricks to the current model after you supply the number of bricks it should use for width and height.

 - Copy orientation
     This macro will apply the orientation of the first selected item to all items in the selection.

 - Generate BOM list
     This macro will format a list of used parts in the current model also know as the model's BOM.


 This script also holds a number of generic animations, namely:
 - Color fest
      This animation will randomly change the color of bricks in the current model for a couple of seconds.

 - Camera test
      This animation will rotate the current model using a third person camera.

 - Explode
      This animation will move all bricks in the current model outwards over time, resulting in an explosive view of the model.

 - Build
      This animation will drop all bricks in the current model from above one by one simulating the building process.

--]]

--import some handy toos/wrapper functions.
local genTools=require('genTools')


--Hello world macro========================================================================================================================
function runHelloWorld()
  --This function will be called when the user clicks the 'Hello' macro in a menu.
  -- So lets perform its groundbreaking task.
  ldc.dialog.runMessage('Hello, welcome to '..ldc.getVersion()..' running lua '..ldc.getLuaVersion())
end


function onParadeStart()
  -- Session -> Animation -> OppenGL animation export
  --300x300
  --FFFFF

  sf = ldc.subfile() -- get reference to ldraw files with parts list
  pieceCounter = 1 -- counter to loop through parts list
  cameraLoopCounter = 1
  frameCounter = 0 -- increments camera angle
  
  myFPS = 8 
  cameraAngleIncrement = 360/myFPS

  unviewPosition = ldc.vector()
  unviewPosition:set(10000,10000,10000)
  viewPosition=ldc.vector()
  viewPosition:set(25,25,25)

  -- setup camera
  cam = ldc.view():getCamera()
  camPos = ldc.vector()
  camPos:set(50,50,50)
  camDist = 1350
  cam:setThirdPerson(camPos,camDist,0,0,0)
  cam:apply(0)

  --place pieces out of view
  refCnt = sf:getRefCount()
  for i=1,refCnt do
    sf:getRef(i):setPos(viewPosition)
  end

  sf:getRef(3):setPos(viewPosition)
end

function onParadeFrame()

  frameCounter = frameCounter + 1
  cam = ldc.view():getCamera()

  if cameraLoopCounter == 1 then
    cam:setThirdPerson(camPos,camDist,cameraAngleIncrement*frameCounter,0,0)
    cam:apply(0)
    if frameCounter == myFPS then
      frameCounter = 0
      cameraLoopCounter = 2
    end
  else  
    if cameraLoopCounter == 2 then
      cam:setThirdPerson(camPos, camDist, 0 ,cameraAngleIncrement*frameCounter,0)
      cam:apply(0)
      if frameCounter == myFPS then
        frameCounter = 0
        cameraLoopCounter = 3
      end
    else
      if cameraLoopCounter == 3 then
        cam:setThirdPerson(camPos,camDist,0,0,cameraAngleIncrement*frameCounter)
        cam:apply(0)
        if frameCounter == myFPS then
          frameCounter = 0
          cameraLoopCounter = 4
        end
      else  
        if cameraLoopCounter == 4 then
          cam:setThirdPerson(camPos,camDist,0,cameraAngleIncrement*frameCounter,cameraAngleIncrement*frameCounter)
          cam:apply(0)
          if frameCounter == myFPS then
            frameCounter = 0
            cameraLoopCounter = 5
          end
        else
          if cameraLoopCounter == 5 then
            cam:setThirdPerson(camPos,camDist,0,cameraAngleIncrement*frameCounter,cameraAngleIncrement*frameCounter)
            cam:apply(0)
            if frameCounter == myFPS then
              frameCounter = 0
              cameraLoopCounter = 1
              if pieceCounter == refCnt then
                sf:getRef(pieceCounter):setPos(unviewPosition)
                pieceCounter = 1
                sf:getRef(pieceCounter):setPos(viewPosition)
                cam:setThirdPerson(cam,camdist,0,0,0)
                cam:apply(0)
              else 
                sf:getRef(pieceCounter):setPos(unviewPosition)
                pieceCounter = pieceCounter + 1
                sf:getRef(pieceCounter):setPos(viewPosition)
                cam:setThirdPerson(camPos,camDist,0,0,0)
                cam:apply(0)
              end
            end
          end 
        end 
      end 
    end 
  end 
end
          
          
--Selection circle macro===================================================================================================================
function runSelCircle()

  --Some dummy proof tests.
  local ses=ldc.session()
  if not ses:isLinked() then
    ldc.dialog.runMessage('No active model.')
    return
  end

  local sel=ses:getSelection()
  local cnt=sel:getRefCount()

  -- only usefull if atleast two items are selected.
  if cnt<2 then
    ldc.dialog.runMessage('Atleast two items should be selected.')
    return
  end

  --ask for the radius to use.
  local radius=tonumber(ldc.dialog.runInput('Radius (10-1000)?', '100'))
  if radius==nil then
    return
  end

  if (radius<10) or (radius>1000) then
    ldc.dialog.runMessage('Invalid radius.')
    return
  end

  local doRnd=true --If set to true it will round to 'studs'.

  --use the first item as the circle's center.
  local cen=sel:getRef(1):getPos()

  --reposition the others around it using the given radius.
  for i=2, cnt do
    local ref=sel:getRef(i)
    local pos=ref:getPos()
    local v=(i-2)/(cnt-1)*math.pi*2
    local x=math.sin(v)*radius
    local z=math.cos(v)*radius

    if doRnd then
      x=genTools.round(x/20)*20
      z=genTools.round(z/20)*20
    end

    pos:set(cen:getX()+x, cen:getY(), cen:getZ()+z)
    ref:setPos(pos)
  end
end



--Add wall macro===========================================================================================================================
function runWall()

  --only usefull if a model is active.
  local ses=ldc.session()
  if not ses:isLinked() then
    ldc.dialog.runMessage('No active model.')
    return
  end

  --Ask for the width and height
  local w=tonumber(ldc.dialog.runInput('Number of bricks (1-100)?', '5'))
  if w==nil then
    return
  end

  local h=tonumber(ldc.dialog.runInput('Height (1-30)?', '3'))
  if h==nil then
    return
  end

  if (w<1) or (w>100) or (h<1) or (h>30) then
    ldc.dialog.runMessage('Invalid number(s)')
    return
  end

  --generate the wall into a new selection.
  local sel=ldc.session.getCurrent():getSelection()
  sel:remove()
  local posOri=ldc.matrix()
  local sf=ldc.subfile()
  local xOfs=-0.5*(w*80)
  local alt=0
  for y=0,h-1 do
    for x=0,w-1 do
      posOri:setPos(xOfs+alt+x*80, -y*24, 0)
      sel:add(sf:addNewRef('3001.dat', 4, posOri))
    end

    if alt==0 then
      alt=40
    else
      alt=0
    end
  end
end


--Orientation copy macro===================================================================================================================
function runOriCopy()

  --only usefull if a model is active.
  local ses=ldc.session()
  if not ses:isLinked() then
    ldc.dialog.runMessage('No active model.')
    return
  end

  local sel=ldc.session():getSelection()
  local cnt=sel:getRefCount()

  if cnt==0 then
    ldc.dialog.runMessage('No selection active.')
    return
  end

  if cnt>1 then
    local ori=sel:getRef(1):getOri()

    for i=2,sel:getRefCount() do
      sel:getRef(i):setOri(ori)
    end
  end
end


--Bom content macro========================================================================================================================
function findBomInfo(info, part, col, cnt)

  --Check if a given reference is already in the list
  --  if so return its index

  for i=1, cnt do
    -- We cant compare part like "infp[i].part==part" because lua objects all have their own instances, so compare the linked content instead.
    if (info[i].part:hasSameLink(part)) and (info[i].color==col) then
      return i
    end
  end

  return nil;
end

function collectBomInfo(sf, info, pCol, cnt)

  --Recursively go trough all reference lines in the given subfile
  local refCnt=sf:getRefCount()
  for i=1,refCnt do

    local ref=sf:getRef(i)
    local refTo=ref:getSubfile()
    local col=ref:getColor()

    --Color 16 means the color is 'inherited' from the reference using the subfile.
    if col==16 then
      col=pCol
    end

    --prevent decimal point in output formatting.
    col=math.tointeger(col)

    --Gather unique part/color combinations, and track how many times they are used.
    --  isRealPart is only true for non moved/colour/shortcut content.
    --  Lets also consider generated content (hoses, bands, etc) to be parts.
    if refTo:isRealPart() or refTo:isGenerated() then

      --This is a inventory item
      -- check if the part/color combo is already in the BOM list.
      local idx=findBomInfo(info, refTo, col, cnt)
      if idx~=nil then
        --This is a previously encountered combo, increase its count.
        info[idx].count=info[idx].count+1
      else
        --It's a new combo, register it.
        cnt=cnt+1
        info[cnt]={part=refTo; count=1; color=col}
      end
    else
      --not a part of interest, recursively process the referenced subfile.
      cnt=collectBomInfo(refTo, info, col, cnt)
    end
  end

  return cnt
end

function runBomGen()

  --There must be an active model in the editor
  local sf=ldc.subfile()
  if not sf:isLinked() then
    ldc.dialog.runMessage('No active model.')
    return
  end

  --Follow the LDraw tree of the current model.
  info={}
  cnt=collectBomInfo(ldc.subfile(), info, 16, 0)

  --Format the results into a csv
  local text='part,color,count,description\n'
  for i=1, cnt do

    local info=info[i]
    local line=info.part:getFileName()..','..info.color..','..info.count..','..info.part:getDescription()

    print(line) --output line to console
    text=text..line..'\n' --append to file content
  end

  --Output some info to console
  print()
  print('Note this BOM includes all hidden content and doesn\'t apply buffer exchange rules.');
  print('If you need more preciese content use the export feature on a part bin inventory group.');

  --output all lines to the clipboard.
  ldc.setClipboardText(text)
end



--Camera test animation====================================================================================================================
function onCameraStart()

  --buffer the rest situation.
  local cam=ldc.view():getCamera()
  camPos=cam:getLookAt()
  camDist=cam:getDistance()
end

function onCameraFrame()

  --Rotate a 3rd person camera around the center of the model.
  local ani=ldc.animation.getCurrent()
  local angle=ani:getFrameTime()/ani:getLength()*360

  local cam=ldc.camera()
  cam:setThirdPerson(camPos, camDist, angle+45, 25, 0)
  cam:apply(0)
end



--Color fest animation=====================================================================================================================
function onColorFestFrame()

  --For each frame change loop through all items and apply a random LDraw color between 0 and 15 to them.
  local sf=ldc.subfile()
  local cnt=sf:getRefCount()
  for i=1,cnt do
    sf:getRef(i):setColor(math.random(15))
  end
end



--Explode animation========================================================================================================================
function onExplodeStart()

  --buffer the rest situation.
  pos={}
  sf=ldc.subfile()
  refCnt=sf:getRefCount()
  for i=1,refCnt do
    pos[i]=sf:getRef(i):getPos()
  end
end

function onExplodeFrame()

  --Calculate the new position relative to the center for all bricks using a simple acceleration formula.
  -- do note this assumes the model center is at 0,0,0
  local newPos=ldc.vector()
  local ani=ldc.animation.getCurrent()
  local mul=3*ani:getFrameTime()/ani:getLength()
  for i=1,refCnt do
    newPos:set(pos[i])
    local dir=ldc.vector()-newPos
    newPos:set(newPos-dir*mul*mul)
    sf:getRef(i):setPos(newPos)
  end
end



--Build animation===========================================================================================================================
function onBuildStart()

  local sf=ldc.subfile()

  build={}
  build.refs={}
  build.pos={}
  build.refCnt=sf:getRefCount()

  if build.refCnt>0 then

    build.dropLen=1.5 --5*ldc.animation():getLength()/build.refCnt
    local ref, p, y, minY, maxY=0

    for i=1,build.refCnt do

      ref=sf:getRef(i)
      build.refs[i]=ref

      p=ref:getPos()
      build.pos[i]=p

      local y=p:getY()
      if (i==1) then
        minY=y
        maxY=y
      else
        if y>maxY then
          maxY=y
        end

        if y<minY then
          minY=y
        end
      end
    end

    build.ceiling=minY-1.25*(maxY-minY)
  end
end

function onBuildFrame()

  local ani=ldc.animation()
  local aniLen=ani:getLength()
  local frTime=ani:getFrameTime()

  for i=1, build.refCnt do

    local refTime=(i-1)/build.refCnt*(aniLen-build.dropLen)
    local vis=refTime<frTime
    local ref=build.refs[i]

    ref:setVisible(vis)

    if vis then
      local pos=ldc.vector(build.pos[i])
      if (refTime+build.dropLen)>frTime then
        local dist=build.ceiling-pos:getY()
        dist=(frTime-refTime)*dist/build.dropLen
        pos:setY(build.ceiling-dist)
      end

      ref:setPos(pos)
    end
  end
end



--Register macros and animations===========================================================================================================
function register()

  local macro=ldc.macro('Hello world')
  macro:setHint('A classic hello world example.')
  macro:setEvent('run', 'runHelloWorld')

  macro:register('Selection circle')
  macro:setEvent('run', 'runSelCircle')
  macro:setHint("Places the items in the current selection in circle around the first selected item.")

  macro:register('Add wall')
  macro:setEvent('run', 'runWall')
  macro:setHint("Adds a wall to the current model.")

  macro:register('Copy orientation')
  macro:setEvent('run', 'runOriCopy')
  macro:setHint("Apply the rotation of the first item in the selection to all items in the selection.")

  macro:register('Generate BOM list')
  macro:setEvent('run', 'runBomGen')
  macro:setHint("Generate the 'Bill Of Material' for the current model.")


  --animations
  local ani=ldc.animation('Color fest')
  ani:setEvent('frame', 'onColorFestFrame')

  ani:register('Camera test')
  ani:setEvent('start', 'onCameraStart')
  ani:setEvent('frame', 'onCameraFrame')

  ani:register('Explode')
  ani:setEvent('start', 'onExplodeStart')
  ani:setEvent('frame', 'onExplodeFrame')

  ani:register('Build')
  ani:setLength(30)
  ani:setEvent('start', 'onBuildStart')
  ani:setEvent('frame', 'onBuildFrame')

  ani:register('Parade')
  ani:setEvent('start','onParadeStart')
  ani:setEvent('frame','onParadeFrame')
  ani:setLength(245)



end

register()

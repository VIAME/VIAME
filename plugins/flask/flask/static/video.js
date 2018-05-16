var frame_start, frame_stop, frame_no;
var video_type, auto_play;

function fastBackward() 
{
  updateFrame(frame_start);
}

function stepBackward() 
{
  updateFrame(frame_no - 1);
}

function playPause() 
{
  auto_play = !auto_play;
}

function stepForward() 
{
  updateFrame(frame_no + 1);
}

function fastForward() 
{
  updateFrame(frame_stop);
}

function playVideo()
{
  if (auto_play){
    stepForward();}
  setTimeout(playVideo, 30);
}

function updateFrame(n)
{
  frame_no = n;
  if (frame_no < frame_start)
    frame_no = frame_stop;    
  if (frame_no > frame_stop)
    frame_no = frame_start;

  var text = document.getElementById("frame_no");
  text.innerHTML = frame_no + "/" + frame_stop;
  var image = document.getElementById("frame_img");
  var clip_no = Math.floor(frame_no / 900);
  if (input_file.lastIndexOf(".") == -1)
    image.src = "/file/video/" + input_file + "/clip" + ("000" + clip_no).slice(-3) + "/" + video_type + "/images/" + ('000000' + frame_no).slice(-6) + ".jpg";
  else
    image.src = "/file/video/" + input_file.substring(0, input_file.lastIndexOf(".")) + "/clip" + ("000" + clip_no).slice(-3) + "/" + video_type + "/images/" + ('000000' + frame_no).slice(-6) + ".jpg";
}


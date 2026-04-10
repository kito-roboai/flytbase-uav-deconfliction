"""
Interactive HTML Report Generator
===================================
Canvas-based drone animation (quad-copter icon, spinning rotors, trail, pulsing
conflict zones) + Plotly path/distance/gantt charts + tabbed layout.
"""

from __future__ import annotations

import json
import os
from typing import List

import numpy as np

from .models import DroneMission, DeconflictionResult
from .trajectory import compute_segment_times, sample_trajectory, position_at_time

PRIMARY_COLOR  = "#00d4ff"
SIM_COLORS     = ["#ff6b35", "#7fff6b", "#ffcc00", "#cc66ff",
                  "#ff66aa", "#66ffee", "#ffaa33", "#aaddff"]
CONFLICT_COLOR = "#ff4444"
CHART_BG   = "#0d1117"
PAPER_BG   = "#161b22"
GRID_COLOR = "#21262d"
TEXT_COLOR = "#e6edf3"

# ---------------------------------------------------------------------------
# Canvas animation JavaScript (raw string — no f-string escaping needed)
# ---------------------------------------------------------------------------
_CANVAS_JS = r"""
if (!CanvasRenderingContext2D.prototype.roundRect) {
  CanvasRenderingContext2D.prototype.roundRect = function(x,y,w,h,r){
    this.beginPath();
    this.moveTo(x+r,y);
    this.arcTo(x+w,y,x+w,y+h,r); this.arcTo(x+w,y+h,x,y+h,r);
    this.arcTo(x,y+h,x,y,r);     this.arcTo(x,y,x+w,y,r);
    this.closePath(); return this;
  };
}

(function(){
  var droneData = window.droneData || {};
  var states = {};

  function lerp(a,b,t){ return a+(b-a)*t; }

  function getPosAtTime(pts, t){
    if(!pts||!pts.length) return null;
    var t0=pts[0][2], tn=pts[pts.length-1][2];
    if(t < t0-0.001 || t > tn+0.001) return null;
    var lo=0, hi=pts.length-1;
    while(lo<hi-1){ var mid=(lo+hi)>>1; if(pts[mid][2]<=t) lo=mid; else hi=mid; }
    var p0=pts[lo], p1=lo+1<pts.length?pts[lo+1]:p0;
    var span=p1[2]-p0[2];
    var a=span<1e-9?0:Math.min(1,Math.max(0,(t-p0[2])/span));
    return [lerp(p0[0],p1[0],a), lerp(p0[1],p1[1],a)];
  }

  function getHeading(pts, t){
    var dt=0.4;
    var t0=pts[0][2], tn=pts[pts.length-1][2];
    var pa=getPosAtTime(pts,Math.max(t0,t-dt));
    var pb=getPosAtTime(pts,Math.min(tn,t+dt));
    if(!pa||!pb) return 0;
    var dx=pb[0]-pa[0], dy=pb[1]-pa[1];
    if(Math.abs(dx)+Math.abs(dy)<0.05) return 0;
    return Math.atan2(dx,dy);
  }

  function makeTC(data,w,h,pad){
    var xr=(data.xmax-data.xmin)||1, yr=(data.ymax-data.ymin)||1;
    return function(wx,wy){
      return [pad+(wx-data.xmin)/xr*(w-2*pad),
              h-pad-(wy-data.ymin)/yr*(h-2*pad)];
    };
  }

  function niceStep(range){
    var raw=range/5, mag=Math.pow(10,Math.floor(Math.log10(raw||1)));
    var n=raw/mag;
    return Math.max(1,(n<1.5?1:n<3.5?2:n<7.5?5:10)*mag);
  }

  function drawGrid(ctx,data,w,h,pad,tc){
    var sx=niceStep(data.xmax-data.xmin), sy=niceStep(data.ymax-data.ymin);
    ctx.strokeStyle='#1a2030'; ctx.lineWidth=1;
    for(var x=Math.ceil(data.xmin/sx)*sx; x<=data.xmax+0.001; x+=sx){
      var cx=tc(x,data.ymin)[0];
      if(cx<pad||cx>w-pad) continue;
      ctx.beginPath(); ctx.moveTo(cx,pad); ctx.lineTo(cx,h-pad); ctx.stroke();
      ctx.fillStyle='#3d444d'; ctx.font='10px monospace'; ctx.textAlign='center';
      ctx.fillText(Math.round(x),cx,h-pad+14);
    }
    for(var y=Math.ceil(data.ymin/sy)*sy; y<=data.ymax+0.001; y+=sy){
      var cy=tc(data.xmin,y)[1];
      if(cy<pad||cy>h-pad) continue;
      ctx.beginPath(); ctx.moveTo(pad,cy); ctx.lineTo(w-pad,cy); ctx.stroke();
      ctx.fillStyle='#3d444d'; ctx.font='10px monospace'; ctx.textAlign='right';
      ctx.fillText(Math.round(y),pad-4,cy+4);
    }
    ctx.textAlign='left';
    ctx.fillStyle='#6e7681'; ctx.font='11px Inter,sans-serif';
    ctx.textAlign='center'; ctx.fillText('X (m)',w/2,h-2);
    ctx.save(); ctx.translate(11,h/2); ctx.rotate(-Math.PI/2);
    ctx.fillText('Y (m)',0,0); ctx.restore(); ctx.textAlign='left';
  }

  function drawPaths(ctx,data,tc){
    [data.primary].concat(data.simulated).forEach(function(dr){
      if(!dr.points.length) return;
      ctx.beginPath();
      dr.points.forEach(function(p,i){
        var c=tc(p[0],p[1]); if(i===0) ctx.moveTo(c[0],c[1]); else ctx.lineTo(c[0],c[1]);
      });
      ctx.strokeStyle=dr.color+'28'; ctx.lineWidth=1.5;
      ctx.setLineDash([5,5]); ctx.stroke(); ctx.setLineDash([]);
      var sp=dr.points[0], ep=dr.points[dr.points.length-1];
      var cs=tc(sp[0],sp[1]), ce=tc(ep[0],ep[1]);
      ctx.beginPath(); ctx.arc(cs[0],cs[1],4,0,6.283);
      ctx.fillStyle=dr.color+'60'; ctx.fill();
      ctx.beginPath(); ctx.arc(ce[0],ce[1],5,0,6.283);
      ctx.strokeStyle=dr.color+'60'; ctx.lineWidth=1.5; ctx.stroke();
    });
  }

  function drawConflicts(ctx,data,tc,now){
    data.conflicts.forEach(function(c){
      var cp=tc(c.x,c.y), pulse=0.5+0.5*Math.sin(now/400);
      var g=ctx.createRadialGradient(cp[0],cp[1],3,cp[0],cp[1],44+12*pulse);
      g.addColorStop(0,'rgba(255,68,68,0.45)'); g.addColorStop(1,'rgba(255,68,68,0)');
      ctx.beginPath(); ctx.arc(cp[0],cp[1],44+12*pulse,0,6.283);
      ctx.fillStyle=g; ctx.fill();
      ctx.beginPath(); ctx.arc(cp[0],cp[1],18,0,6.283);
      ctx.strokeStyle='#ff4444'; ctx.lineWidth=2;
      ctx.globalAlpha=0.5+0.5*pulse; ctx.stroke(); ctx.globalAlpha=1;
      ctx.strokeStyle='#ff4444'; ctx.lineWidth=2.5;
      ctx.beginPath();
      ctx.moveTo(cp[0]-9,cp[1]-9); ctx.lineTo(cp[0]+9,cp[1]+9);
      ctx.moveTo(cp[0]+9,cp[1]-9); ctx.lineTo(cp[0]-9,cp[1]+9);
      ctx.stroke();
      ctx.fillStyle='#ff4444'; ctx.font='bold 10px Inter,monospace';
      ctx.textAlign='center'; ctx.fillText('CONFLICT',cp[0],cp[1]+32);
      ctx.textAlign='left';
    });
  }

  function drawTrail(ctx,pts,t,tc,color){
    var dur=3.5, tp=[];
    for(var i=0;i<pts.length;i++){
      if(pts[i][2]>=t-dur && pts[i][2]<=t) tp.push(pts[i]);
    }
    if(tp.length<2) return;
    for(var j=0;j<tp.length-1;j++){
      var a=(j+1)/tp.length*0.8;
      var c1=tc(tp[j][0],tp[j][1]), c2=tc(tp[j+1][0],tp[j+1][1]);
      ctx.beginPath(); ctx.moveTo(c1[0],c1[1]); ctx.lineTo(c2[0],c2[1]);
      ctx.strokeStyle=color; ctx.globalAlpha=a; ctx.lineWidth=2.5;
      ctx.lineCap='round'; ctx.stroke();
    }
    ctx.globalAlpha=1;
  }

  function drawDrone(ctx,cx,cy,angle,color,sz,label,now){
    ctx.save(); ctx.translate(cx,cy); ctx.rotate(angle);
    var arm=sz, rr=sz*0.40, br=sz*0.22;
    var spin=(now*0.015)%(2*Math.PI);

    [45,135,225,315].forEach(function(deg){
      var rad=deg*Math.PI/180;
      ctx.beginPath(); ctx.moveTo(0,0);
      ctx.lineTo(Math.cos(rad)*arm, Math.sin(rad)*arm);
      ctx.strokeStyle=color; ctx.lineWidth=sz*0.14; ctx.lineCap='round'; ctx.stroke();
    });

    [45,135,225,315].forEach(function(deg){
      var rad=deg*Math.PI/180;
      var rx=Math.cos(rad)*arm, ry=Math.sin(rad)*arm;
      var g=ctx.createRadialGradient(rx,ry,0,rx,ry,rr);
      g.addColorStop(0,color+'cc'); g.addColorStop(0.6,color+'33'); g.addColorStop(1,color+'00');
      ctx.beginPath(); ctx.arc(rx,ry,rr,0,6.283); ctx.fillStyle=g; ctx.fill();
      ctx.save(); ctx.translate(rx,ry); ctx.rotate(spin);
      ctx.strokeStyle=color+'cc'; ctx.lineWidth=rr*0.22; ctx.lineCap='round';
      for(var b=0;b<2;b++){
        ctx.beginPath(); ctx.moveTo(-rr*0.85,0); ctx.lineTo(rr*0.85,0); ctx.stroke();
        ctx.rotate(Math.PI/2);
      }
      ctx.restore();
      ctx.beginPath(); ctx.arc(rx,ry,rr*0.17,0,6.283); ctx.fillStyle=color; ctx.fill();
    });

    var bg=ctx.createRadialGradient(0,0,0,0,0,br*3);
    bg.addColorStop(0,color+'55'); bg.addColorStop(1,color+'00');
    ctx.beginPath(); ctx.arc(0,0,br*3,0,6.283); ctx.fillStyle=bg; ctx.fill();
    ctx.beginPath(); ctx.arc(0,0,br,0,6.283); ctx.fillStyle=color; ctx.fill();
    ctx.beginPath(); ctx.arc(0,-br*0.55,br*0.28,0,6.283); ctx.fillStyle='#fff'; ctx.fill();

    ctx.rotate(-angle);
    ctx.font='bold '+Math.round(sz*0.62)+'px Inter,monospace';
    ctx.fillStyle=color; ctx.textAlign='center';
    ctx.fillText(label,0,-arm-5);
    ctx.restore();
  }

  function drawLegend(ctx,data,W,H){
    var allDr=[data.primary].concat(data.simulated);
    var lh=18,pad=8,maxW=0;
    ctx.font='11px Inter,monospace';
    allDr.forEach(function(dr){var tw=ctx.measureText(dr.id).width;if(tw>maxW)maxW=tw;});
    var bw=maxW+32,bh=lh*allDr.length+pad*2,lx=12,ly=12;
    ctx.fillStyle='rgba(13,17,23,0.85)';ctx.roundRect(lx,ly,bw,bh,5);ctx.fill();
    allDr.forEach(function(dr,i){
      var y=ly+pad+i*lh+lh*0.55;
      ctx.beginPath();ctx.arc(lx+pad+5,y,4,0,6.283);ctx.fillStyle=dr.color;ctx.fill();
      ctx.fillStyle=dr.color;ctx.font='11px Inter,monospace';ctx.textAlign='left';
      ctx.fillText(dr.id,lx+pad+14,y+4);
    });
    ctx.textAlign='left';
  }

  function drawSeparation(ctx,data,t,tc){
    var ppos=getPosAtTime(data.primary.points,t);if(!ppos)return;
    data.simulated.forEach(function(sim){
      var spos=getPosAtTime(sim.points,t);if(!spos)return;
      var dx=ppos[0]-spos[0],dy=ppos[1]-spos[1];
      var dist=Math.sqrt(dx*dx+dy*dy);
      var cp=tc(ppos[0],ppos[1]),cs=tc(spos[0],spos[1]);
      var mx=(cp[0]+cs[0])/2,my=(cp[1]+cs[1])/2;
      var isConf=data.safety_buffer>0&&dist<data.safety_buffer;
      var col=isConf?'#ff4444':'#6e7681';
      ctx.beginPath();ctx.moveTo(cp[0],cp[1]);ctx.lineTo(cs[0],cs[1]);
      ctx.strokeStyle=col+'55';ctx.lineWidth=1;ctx.setLineDash([3,4]);ctx.stroke();ctx.setLineDash([]);
      ctx.fillStyle='rgba(13,17,23,0.75)';ctx.font='bold 10px Inter,monospace';
      ctx.textAlign='center';
      var tw=ctx.measureText(dist.toFixed(1)+'m').width;
      ctx.fillRect(mx-tw/2-3,my-16,tw+6,14);
      ctx.fillStyle=col;ctx.fillText(dist.toFixed(1)+'m',mx,my-5);
      ctx.textAlign='left';
    });
  }

  function drawCountdown(ctx,data,t,W){
    if(!data.conflicts||!data.conflicts.length)return;
    var next=null;
    data.conflicts.forEach(function(c){if(c.t>t&&(next===null||c.t<next.t))next=c;});
    if(!next||next.t-t>20)return;
    var dt=next.t-t;
    var pulse=0.5+0.5*Math.sin(Date.now()/280);
    ctx.fillStyle='rgba(20,5,5,0.88)';ctx.roundRect(W/2-100,12,200,28,5);ctx.fill();
    var alpha=Math.round((180+75*pulse)).toString(16).padStart(2,'0');
    ctx.fillStyle='#ff4444'+alpha;ctx.font='bold 12px Inter,monospace';ctx.textAlign='center';
    ctx.fillText('\u26a0  Conflict in '+dt.toFixed(1)+'s',W/2,31);ctx.textAlign='left';
  }

  function initCanvas(si){
    var data=window.droneData[si];
    if(!data) return;
    var canvas=document.getElementById('drone-canvas-'+si);
    if(!canvas) return;
    canvas.width=canvas.parentElement.clientWidth||820;
    canvas.height=480;
    var W=canvas.width, H=canvas.height, PAD=55;
    var tc=makeTC(data,W,H,PAD);
    var st=states[si]={t:data.tmin,playing:false,speed:1,animId:null,lastTs:null};
    var slider=document.getElementById('time-slider-'+si);
    var lbl=document.getElementById('time-display-'+si);

    function draw(now){
      now=now||performance.now();
      W=canvas.width; H=canvas.height; tc=makeTC(data,W,H,PAD);
      var ctx=canvas.getContext('2d');
      ctx.fillStyle='#0d1117'; ctx.fillRect(0,0,W,H);
      drawGrid(ctx,data,W,H,PAD,tc);
      drawPaths(ctx,data,tc);
      drawConflicts(ctx,data,tc,now);
      var allDr=[data.primary].concat(data.simulated);
      var dsz=Math.max(11,Math.min(20,Math.min(W,H)/32));
      allDr.forEach(function(dr){
        if(!dr.points.length) return;
        var t0=dr.points[0][2], t1=dr.points[dr.points.length-1][2];
        if(st.t<t0-0.001||st.t>t1+0.001) return;
        var pos=getPosAtTime(dr.points,st.t); if(!pos) return;
        var hdg=getHeading(dr.points,st.t);
        var cp=tc(pos[0],pos[1]);
        drawTrail(ctx,dr.points,st.t,tc,dr.color);
        drawDrone(ctx,cp[0],cp[1],hdg,dr.color,dsz,dr.id,now);
      });
      drawSeparation(ctx,data,st.t,tc);
      drawLegend(ctx,data,W,H);
      ctx.fillStyle='rgba(13,17,23,0.82)';
      ctx.roundRect(W-130,12,118,28,6); ctx.fill();
      ctx.fillStyle='#00d4ff'; ctx.font='13px Inter,monospace'; ctx.textAlign='right';
      ctx.fillText('t = '+st.t.toFixed(1)+' s',W-16,31); ctx.textAlign='left';
      drawCountdown(ctx,data,st.t,W);
      var pct=(data.tmax-data.tmin)>0?(st.t-data.tmin)/(data.tmax-data.tmin):0;
      if(slider) slider.value=Math.round(pct*1000);
      if(lbl) lbl.textContent='t = '+st.t.toFixed(1)+' s';
    }

    function step(ts){
      if(st.lastTs!==null){
        var dt=(ts-st.lastTs)/1000;
        st.t+=dt*st.speed*(data.tmax-data.tmin)/8;
        if(st.t>=data.tmax){st.t=data.tmin;} // auto-loop
      }
      st.lastTs=ts; draw(ts);
      if(st.playing) st.animId=requestAnimationFrame(step);
    }

    var pBtn=document.getElementById('ctrl-play-'+si);
    var pauseBtn=document.getElementById('ctrl-pause-'+si);
    var resetBtn=document.getElementById('ctrl-reset-'+si);

    if(pBtn) pBtn.onclick=function(){
      if(st.t>=data.tmax-0.01) st.t=data.tmin;
      st.playing=true; st.lastTs=null; st.animId=requestAnimationFrame(step);
    };
    if(pauseBtn) pauseBtn.onclick=function(){
      st.playing=false; if(st.animId){cancelAnimationFrame(st.animId);st.animId=null;}
    };
    if(resetBtn) resetBtn.onclick=function(){
      st.playing=false; if(st.animId){cancelAnimationFrame(st.animId);st.animId=null;}
      st.t=data.tmin; draw();
    };

    document.querySelectorAll('.speed-btn[data-si="'+si+'"]').forEach(function(btn){
      btn.onclick=function(){
        st.speed=parseFloat(this.dataset.speed);
        document.querySelectorAll('.speed-btn[data-si="'+si+'"]').forEach(function(b){
          b.classList.toggle('speed-active',b===btn);
        });
      };
    });

    if(slider) slider.oninput=function(){
      st.t=data.tmin+(data.tmax-data.tmin)*this.value/1000;
      if(!st.playing) draw();
    };

    if(window.ResizeObserver){
      new ResizeObserver(function(){
        canvas.width=canvas.parentElement.clientWidth||820;
        if(!st.playing) draw();
      }).observe(canvas.parentElement);
    }

    draw();
  }

  window.__reinitCanvas=function(si){
    var old=states[si];
    if(old&&old.animId) cancelAnimationFrame(old.animId);
    initCanvas(si);
  };

  window.addEventListener('DOMContentLoaded',function(){
    Object.keys(window.droneData).forEach(function(si){
      var d=window.droneData[si];
      if(d&&!d.is3d){
        var tab=document.getElementById('canvas-'+si);
        if(tab&&tab.classList.contains('tab-active')) initCanvas(parseInt(si));
      }
    });
  });
})();
"""

# ---------------------------------------------------------------------------
# 3D isometric canvas animation JavaScript (raw string — no f-string escaping)
# ---------------------------------------------------------------------------
_CANVAS_3D_JS = r"""
(function(){
  var _prev2d=window.__reinitCanvas;
  var states3={};

  function lerp(a,b,t){return a+(b-a)*t;}

  function getPt3(pts,t){
    if(!pts||!pts.length)return null;
    var t0=pts[0][3],tn=pts[pts.length-1][3];
    if(t<t0-0.001||t>tn+0.001)return null;
    var lo=0,hi=pts.length-1;
    while(lo<hi-1){var mid=(lo+hi)>>1;if(pts[mid][3]<=t)lo=mid;else hi=mid;}
    var p0=pts[lo],p1=lo+1<pts.length?pts[lo+1]:p0;
    var sp=p1[3]-p0[3];
    var a=sp<1e-9?0:Math.min(1,Math.max(0,(t-p0[3])/sp));
    return[lerp(p0[0],p1[0],a),lerp(p0[1],p1[1],a),lerp(p0[2],p1[2],a)];
  }

  function hdg3(pts,t){
    var dt=0.4,t0=pts[0][3],tn=pts[pts.length-1][3];
    var pa=getPt3(pts,Math.max(t0,t-dt)),pb=getPt3(pts,Math.min(tn,t+dt));
    if(!pa||!pb)return 0;
    var dx=pb[0]-pa[0],dy=pb[1]-pa[1];
    if(Math.abs(dx)+Math.abs(dy)<0.05)return 0;
    return Math.atan2(dx,dy);
  }

  function mkProj(data,W,H,az){
    var cx=(data.xmin+data.xmax)/2,cy=(data.ymin+data.ymax)/2;
    var hz=Math.max(data.xmax-data.xmin,data.ymax-data.ymin,1);
    var vz=Math.max(data.zmax-data.zmin,1);
    var PAD=70;
    var sc=Math.min((W-PAD*2)/hz,(H*0.52)/hz);
    var scZ=Math.min(sc*1.1,(H*0.38)/vz);
    var baseY=H*0.64;
    return function(wx,wy,wz){
      var x=wx-cx,y=wy-cy,z=(wz||0)-data.zmin;
      var rx=x*Math.cos(az)-y*Math.sin(az);
      var ry=x*Math.sin(az)+y*Math.cos(az);
      return[W/2+rx*sc,baseY-z*scZ+ry*0.45*sc];
    };
  }

  function niceS3(range){
    var raw=range/5,mag=Math.pow(10,Math.floor(Math.log10(raw||1)));
    var n=raw/mag;
    return Math.max(1,(n<1.5?1:n<3.5?2:n<7.5?5:10)*mag);
  }

  function drawGrid3(ctx,data,proj){
    var z0=data.zmin;
    var xs=niceS3(data.xmax-data.xmin),ys=niceS3(data.ymax-data.ymin);
    ctx.strokeStyle='#1a2030';ctx.lineWidth=0.8;
    for(var x=Math.ceil(data.xmin/xs)*xs;x<=data.xmax+0.01;x+=xs){
      var a=proj(x,data.ymin,z0),b=proj(x,data.ymax,z0);
      ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();
      ctx.fillStyle='#3d444d';ctx.font='10px monospace';ctx.textAlign='center';
      ctx.fillText(Math.round(x),(a[0]+b[0])/2,(a[1]+b[1])/2+14);
    }
    for(var y=Math.ceil(data.ymin/ys)*ys;y<=data.ymax+0.01;y+=ys){
      var a=proj(data.xmin,y,z0),b=proj(data.xmax,y,z0);
      ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();
      ctx.fillStyle='#3d444d';ctx.font='10px monospace';ctx.textAlign='right';
      ctx.fillText(Math.round(y),a[0]-4,a[1]+4);
    }
    ctx.textAlign='left';
    ctx.strokeStyle='#21262d';ctx.lineWidth=1;
    var corners=[[data.xmin,data.ymin],[data.xmax,data.ymin],[data.xmax,data.ymax],[data.xmin,data.ymax]];
    ctx.beginPath();
    corners.forEach(function(c,i){var p=proj(c[0],c[1],z0);if(i===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1]);});
    ctx.closePath();ctx.stroke();
    // Z-axis line at corner
    var zc=proj(data.xmin,data.ymin,z0),zt=proj(data.xmin,data.ymin,data.zmax);
    ctx.beginPath();ctx.moveTo(zc[0],zc[1]);ctx.lineTo(zt[0],zt[1]);
    ctx.strokeStyle='#2a3040';ctx.lineWidth=1;ctx.stroke();
    var zs=niceS3(data.zmax-data.zmin);
    for(var z=Math.ceil(data.zmin/zs)*zs;z<=data.zmax+0.01;z+=zs){
      var tp=proj(data.xmin,data.ymin,z);
      ctx.beginPath();ctx.moveTo(tp[0]-4,tp[1]);ctx.lineTo(tp[0]+4,tp[1]);
      ctx.strokeStyle='#3d444d';ctx.lineWidth=1;ctx.stroke();
      ctx.fillStyle='#3d444d';ctx.font='10px monospace';ctx.textAlign='right';
      ctx.fillText(Math.round(z)+'m',tp[0]-6,tp[1]+4);
    }
    ctx.textAlign='left';
  }

  function drawPaths3(ctx,data,proj){
    [data.primary].concat(data.simulated).forEach(function(dr){
      if(!dr.points.length)return;
      ctx.beginPath();
      dr.points.forEach(function(p,i){
        var c=proj(p[0],p[1],p[2]);
        if(i===0)ctx.moveTo(c[0],c[1]);else ctx.lineTo(c[0],c[1]);
      });
      ctx.strokeStyle=dr.color+'30';ctx.lineWidth=1.5;
      ctx.setLineDash([4,4]);ctx.stroke();ctx.setLineDash([]);
      // start/end markers
      var sp=dr.points[0],ep=dr.points[dr.points.length-1];
      var cs=proj(sp[0],sp[1],sp[2]),ce=proj(ep[0],ep[1],ep[2]);
      ctx.beginPath();ctx.arc(cs[0],cs[1],4,0,6.283);ctx.fillStyle=dr.color+'50';ctx.fill();
      ctx.beginPath();ctx.arc(ce[0],ce[1],5,0,6.283);ctx.strokeStyle=dr.color+'50';ctx.lineWidth=1.5;ctx.stroke();
    });
  }

  function drawTrail3(ctx,pts,t,proj,color){
    var dur=3.5,tp=[];
    for(var i=0;i<pts.length;i++){if(pts[i][3]>=t-dur&&pts[i][3]<=t)tp.push(pts[i]);}
    if(tp.length<2)return;
    for(var j=0;j<tp.length-1;j++){
      var a=(j+1)/tp.length*0.8;
      var c1=proj(tp[j][0],tp[j][1],tp[j][2]),c2=proj(tp[j+1][0],tp[j+1][1],tp[j+1][2]);
      ctx.beginPath();ctx.moveTo(c1[0],c1[1]);ctx.lineTo(c2[0],c2[1]);
      ctx.strokeStyle=color;ctx.globalAlpha=a;ctx.lineWidth=2.5;
      ctx.lineCap='round';ctx.stroke();
    }
    ctx.globalAlpha=1;
  }

  function drawShadow3(ctx,data,proj,wx,wy,wz,color){
    // Only ground circle — no vertical drop-line to avoid visual clutter
    var gnd=proj(wx,wy,data.zmin);
    ctx.beginPath();ctx.arc(gnd[0],gnd[1],4,0,6.283);
    ctx.fillStyle=color+'40';ctx.fill();
  }

  function drawConflicts3(ctx,data,proj,now){
    data.conflicts.forEach(function(c){
      var cp=proj(c.x,c.y,c.z!==undefined?c.z:data.zmin);
      var pulse=0.5+0.5*Math.sin(now/400);
      var g=ctx.createRadialGradient(cp[0],cp[1],2,cp[0],cp[1],36+10*pulse);
      g.addColorStop(0,'rgba(255,68,68,0.5)');g.addColorStop(1,'rgba(255,68,68,0)');
      ctx.beginPath();ctx.arc(cp[0],cp[1],36+10*pulse,0,6.283);ctx.fillStyle=g;ctx.fill();
      ctx.beginPath();ctx.arc(cp[0],cp[1],14,0,6.283);
      ctx.strokeStyle='#ff4444';ctx.lineWidth=2;
      ctx.globalAlpha=0.5+0.5*pulse;ctx.stroke();ctx.globalAlpha=1;
      ctx.strokeStyle='#ff4444';ctx.lineWidth=2;
      ctx.beginPath();ctx.moveTo(cp[0]-8,cp[1]-8);ctx.lineTo(cp[0]+8,cp[1]+8);
      ctx.moveTo(cp[0]+8,cp[1]-8);ctx.lineTo(cp[0]-8,cp[1]+8);ctx.stroke();
      ctx.fillStyle='#ff4444';ctx.font='bold 10px Inter,monospace';
      ctx.textAlign='center';ctx.fillText('CONFLICT',cp[0],cp[1]+30);ctx.textAlign='left';
    });
  }

  function drawDrone3d(ctx,cx,cy,angle,color,sz,label,now){
    ctx.save();ctx.translate(cx,cy);ctx.rotate(angle);
    var arm=sz,rr=sz*0.40,br=sz*0.22;
    var spin=(now*0.015)%(2*Math.PI);
    [45,135,225,315].forEach(function(deg){
      var rad=deg*Math.PI/180;
      ctx.beginPath();ctx.moveTo(0,0);ctx.lineTo(Math.cos(rad)*arm,Math.sin(rad)*arm);
      ctx.strokeStyle=color;ctx.lineWidth=sz*0.14;ctx.lineCap='round';ctx.stroke();
    });
    [45,135,225,315].forEach(function(deg){
      var rad=deg*Math.PI/180;
      var rx=Math.cos(rad)*arm,ry=Math.sin(rad)*arm;
      var g=ctx.createRadialGradient(rx,ry,0,rx,ry,rr);
      g.addColorStop(0,color+'cc');g.addColorStop(0.6,color+'33');g.addColorStop(1,color+'00');
      ctx.beginPath();ctx.arc(rx,ry,rr,0,6.283);ctx.fillStyle=g;ctx.fill();
      ctx.save();ctx.translate(rx,ry);ctx.rotate(spin);
      ctx.strokeStyle=color+'cc';ctx.lineWidth=rr*0.22;ctx.lineCap='round';
      for(var b=0;b<2;b++){
        ctx.beginPath();ctx.moveTo(-rr*0.85,0);ctx.lineTo(rr*0.85,0);ctx.stroke();ctx.rotate(Math.PI/2);
      }
      ctx.restore();
      ctx.beginPath();ctx.arc(rx,ry,rr*0.17,0,6.283);ctx.fillStyle=color;ctx.fill();
    });
    var bg=ctx.createRadialGradient(0,0,0,0,0,br*3);
    bg.addColorStop(0,color+'55');bg.addColorStop(1,color+'00');
    ctx.beginPath();ctx.arc(0,0,br*3,0,6.283);ctx.fillStyle=bg;ctx.fill();
    ctx.beginPath();ctx.arc(0,0,br,0,6.283);ctx.fillStyle=color;ctx.fill();
    ctx.beginPath();ctx.arc(0,-br*0.55,br*0.28,0,6.283);ctx.fillStyle='#fff';ctx.fill();
    ctx.rotate(-angle);
    ctx.font='bold '+Math.round(sz*0.62)+'px Inter,monospace';
    ctx.fillStyle=color;ctx.textAlign='center';
    ctx.fillText(label,0,-arm-5);
    ctx.restore();
  }

  function initCanvas3d(si){
    var data=window.droneData[si];
    if(!data||!data.is3d)return;
    var canvas=document.getElementById('drone-canvas-'+si);
    if(!canvas)return;
    canvas.width=canvas.parentElement.clientWidth||820;
    canvas.height=500;
    var st=states3[si]={t:data.tmin,playing:false,speed:1,animId:null,lastTs:null,az:-0.65,dragging:false,dragX:0};

    function pr(){return mkProj(data,canvas.width,canvas.height,st.az);}

    function draw(now){
      now=now||performance.now();
      var W=canvas.width,H=canvas.height;
      var ctx=canvas.getContext('2d'),proj=pr();
      ctx.fillStyle='#0d1117';ctx.fillRect(0,0,W,H);
      drawGrid3(ctx,data,proj);
      drawPaths3(ctx,data,proj);
      drawConflicts3(ctx,data,proj,now);
      var allDr=[data.primary].concat(data.simulated);
      var dsz=Math.max(11,Math.min(20,Math.min(W,H)/32));
      allDr.forEach(function(dr){
        if(!dr.points.length)return;
        var t0=dr.points[0][3],t1=dr.points[dr.points.length-1][3];
        if(st.t<t0-0.001||st.t>t1+0.001)return;
        var pos=getPt3(dr.points,st.t);if(!pos)return;
        var hdg=hdg3(dr.points,st.t);
        drawTrail3(ctx,dr.points,st.t,proj,dr.color);
        drawShadow3(ctx,data,proj,pos[0],pos[1],pos[2],dr.color);
        var cp=proj(pos[0],pos[1],pos[2]);
        drawDrone3d(ctx,cp[0],cp[1],hdg,dr.color,dsz,dr.id,now);
        ctx.fillStyle=dr.color+'cc';ctx.font='bold 10px monospace';ctx.textAlign='center';
        ctx.fillText('z='+pos[2].toFixed(0)+'m',cp[0],cp[1]+dsz+16);ctx.textAlign='left';
      });
      // time display
      ctx.fillStyle='rgba(13,17,23,0.82)';
      ctx.roundRect(W-130,12,118,28,6);ctx.fill();
      ctx.fillStyle='#00d4ff';ctx.font='13px Inter,monospace';ctx.textAlign='right';
      ctx.fillText('t = '+st.t.toFixed(1)+' s',W-16,31);ctx.textAlign='left';
      // hint
      ctx.fillStyle='#3d444d';ctx.font='11px Inter,sans-serif';ctx.textAlign='center';
      ctx.fillText('Mouse drag \u2194 to rotate',W/2,H-8);ctx.textAlign='left';
      // color legend top-left
      var lgDr=allDr,lh3=18,pad3=8,maxW3=0;
      ctx.font='11px Inter,monospace';
      lgDr.forEach(function(dr){var tw=ctx.measureText(dr.id).width;if(tw>maxW3)maxW3=tw;});
      var bw3=maxW3+32,bh3=lh3*lgDr.length+pad3*2,lx3=12,ly3=12;
      ctx.fillStyle='rgba(13,17,23,0.85)';ctx.roundRect(lx3,ly3,bw3,bh3,5);ctx.fill();
      lgDr.forEach(function(dr,i){
        var yl=ly3+pad3+i*lh3+lh3*0.55;
        ctx.beginPath();ctx.arc(lx3+pad3+5,yl,4,0,6.283);ctx.fillStyle=dr.color;ctx.fill();
        ctx.fillStyle=dr.color;ctx.font='11px Inter,monospace';ctx.textAlign='left';
        ctx.fillText(dr.id,lx3+pad3+14,yl+4);
      });
      ctx.textAlign='left';
      // alt legend bottom-left
      var lx=14,ly=H-28;
      ctx.fillStyle='#6e7681';ctx.font='11px monospace';ctx.fillText('Alt:',lx,ly);
      allDr.forEach(function(dr,i){
        if(!dr.points.length)return;
        var clamp=Math.max(dr.points[0][3],Math.min(dr.points[dr.points.length-1][3],st.t));
        var pos=getPt3(dr.points,clamp);if(!pos)return;
        ctx.fillStyle=dr.color;
        ctx.fillText(dr.id+' \u2192 '+pos[2].toFixed(0)+'m',lx+34+i*100,ly);
      });
      // conflict countdown
      if(data.conflicts&&data.conflicts.length){
        var nxt=null;
        data.conflicts.forEach(function(c){if(c.t>st.t&&(nxt===null||c.t<nxt.t))nxt=c;});
        if(nxt&&nxt.t-st.t<=20){
          var dtt=nxt.t-st.t,pul=0.5+0.5*Math.sin(Date.now()/280);
          ctx.fillStyle='rgba(20,5,5,0.88)';ctx.roundRect(W/2-100,12,200,28,5);ctx.fill();
          var alp=Math.round(180+75*pul).toString(16).padStart(2,'0');
          ctx.fillStyle='#ff4444'+alp;ctx.font='bold 12px Inter,monospace';ctx.textAlign='center';
          ctx.fillText('\u26a0  Conflict in '+dtt.toFixed(1)+'s',W/2,31);ctx.textAlign='left';
        }
      }
      // slider + label update
      var slider=document.getElementById('time-slider-'+si);
      var lbl=document.getElementById('time-display-'+si);
      var pct=(data.tmax-data.tmin)>0?(st.t-data.tmin)/(data.tmax-data.tmin):0;
      if(slider)slider.value=Math.round(pct*1000);
      if(lbl)lbl.textContent='t = '+st.t.toFixed(1)+' s';
    }

    function step(ts){
      if(st.lastTs!==null){
        var dt=(ts-st.lastTs)/1000;
        st.t+=dt*st.speed*(data.tmax-data.tmin)/8;
        if(st.t>=data.tmax){st.t=data.tmin;} // auto-loop
      }
      st.lastTs=ts;draw(ts);
      if(st.playing)st.animId=requestAnimationFrame(step);
    }

    // Drag-to-rotate
    canvas.style.cursor='grab';
    canvas.onmousedown=function(e){st.dragging=true;st.dragX=e.clientX;canvas.style.cursor='grabbing';};
    canvas.onmousemove=function(e){
      if(!st.dragging)return;
      st.az+=(e.clientX-st.dragX)*0.009;st.dragX=e.clientX;
      if(!st.playing)draw();
    };
    canvas.onmouseup=canvas.onmouseleave=function(){st.dragging=false;canvas.style.cursor='grab';};
    canvas.ontouchstart=function(e){st.dragging=true;st.dragX=e.touches[0].clientX;e.preventDefault();};
    canvas.ontouchmove=function(e){
      if(!st.dragging)return;
      st.az+=(e.touches[0].clientX-st.dragX)*0.009;st.dragX=e.touches[0].clientX;
      if(!st.playing)draw();e.preventDefault();
    };
    canvas.ontouchend=function(){st.dragging=false;};

    // Controls
    var pBtn=document.getElementById('ctrl-play-'+si);
    var pauseBtn=document.getElementById('ctrl-pause-'+si);
    var resetBtn=document.getElementById('ctrl-reset-'+si);
    if(pBtn)pBtn.onclick=function(){
      if(st.t>=data.tmax-0.01)st.t=data.tmin;
      st.playing=true;st.lastTs=null;st.animId=requestAnimationFrame(step);
    };
    if(pauseBtn)pauseBtn.onclick=function(){
      st.playing=false;if(st.animId){cancelAnimationFrame(st.animId);st.animId=null;}
    };
    if(resetBtn)resetBtn.onclick=function(){
      st.playing=false;if(st.animId){cancelAnimationFrame(st.animId);st.animId=null;}
      st.t=data.tmin;draw();
    };
    document.querySelectorAll('.speed-btn[data-si="'+si+'"]').forEach(function(btn){
      btn.onclick=function(){
        st.speed=parseFloat(this.dataset.speed);
        document.querySelectorAll('.speed-btn[data-si="'+si+'"]').forEach(function(b){
          b.classList.toggle('speed-active',b===btn);
        });
      };
    });
    var slider=document.getElementById('time-slider-'+si);
    if(slider)slider.oninput=function(){
      st.t=data.tmin+(data.tmax-data.tmin)*this.value/1000;
      if(!st.playing)draw();
    };
    if(window.ResizeObserver){
      new ResizeObserver(function(){
        canvas.width=canvas.parentElement.clientWidth||820;
        if(!st.playing)draw();
      }).observe(canvas.parentElement);
    }
    draw();
  }

  // Override __reinitCanvas to dispatch 2D vs 3D
  window.__reinitCanvas=function(si){
    var data=window.droneData[si];
    if(data&&data.is3d){
      var old=states3[si];if(old&&old.animId)cancelAnimationFrame(old.animId);
      initCanvas3d(si);
    } else if(_prev2d){
      _prev2d(si);
    }
  };

  window.addEventListener('DOMContentLoaded',function(){
    Object.keys(window.droneData).forEach(function(si){
      var d=window.droneData[si];
      if(d&&d.is3d){
        var tab=document.getElementById('canvas-'+si);
        if(tab&&tab.classList.contains('tab-active'))initCanvas3d(parseInt(si));
      }
    });
  });
})();
"""


# ---------------------------------------------------------------------------
# Helpers to build canvas trajectory data
# ---------------------------------------------------------------------------

def _build_canvas_data(
    primary: DroneMission,
    simulated: List[DroneMission],
    result: DeconflictionResult,
    is_3d: bool = False,
    safety_buffer: float = 5.0,
) -> dict:
    N = 400
    seg_p = compute_segment_times(primary)
    t0_p, t1_p = seg_p[0][0], seg_p[-1][1]
    all_t0, all_t1 = [t0_p], [t1_p]
    for sim in simulated:
        st = compute_segment_times(sim)
        all_t0.append(st[0][0]); all_t1.append(st[-1][1])
    tmin, tmax = float(min(all_t0)), float(max(all_t1))
    t_vals = np.linspace(tmin, tmax, N)

    def sample(mission):
        seg = compute_segment_times(mission)
        mt0, mt1 = seg[0][0], seg[-1][1]
        pts = []
        for t in t_vals:
            if t < mt0 or t > mt1:
                continue
            pos = position_at_time(mission, t)
            if pos is not None:
                if is_3d:
                    pts.append([round(float(pos[0]), 2),
                                 round(float(pos[1]), 2),
                                 round(float(pos[2]), 2),
                                 round(float(t), 3)])
                else:
                    pts.append([round(float(pos[0]), 2),
                                 round(float(pos[1]), 2),
                                 round(float(t), 3)])
        return pts

    p_pts = sample(primary)
    s_pts  = [sample(sim) for sim in simulated]

    all_x = [p[0] for p in p_pts] + [p[0] for sp in s_pts for p in sp] or [0.0, 100.0]
    all_y = [p[1] for p in p_pts] + [p[1] for sp in s_pts for p in sp] or [0.0, 100.0]
    mx = max((max(all_x) - min(all_x)) * 0.15, 15)
    my = max((max(all_y) - min(all_y)) * 0.15, 15)

    d = {
        "primary": {"id": primary.drone_id, "points": p_pts, "color": PRIMARY_COLOR},
        "simulated": [
            {"id": sim.drone_id, "points": s_pts[k],
             "color": SIM_COLORS[k % len(SIM_COLORS)]}
            for k, sim in enumerate(simulated)
        ],
        "conflicts": [
            {"x": round(float(c.location[0]), 2),
             "y": round(float(c.location[1]), 2),
             "z": round(float(c.location[2]), 2) if is_3d else 0.0,
             "t": round(float(c.time), 3),
             "sep": round(float(c.separation), 3)}
            for c in result.conflicts
        ],
        "tmin": round(tmin, 3), "tmax": round(tmax, 3),
        "xmin": round(min(all_x) - mx, 1), "xmax": round(max(all_x) + mx, 1),
        "ymin": round(min(all_y) - my, 1), "ymax": round(max(all_y) + my, 1),
        "safety_buffer": round(float(safety_buffer), 2),
    }
    if is_3d:
        all_z = [p[2] for p in p_pts] + [p[2] for sp in s_pts for p in sp] or [0.0, 50.0]
        d["zmin"] = 0.0
        d["zmax"] = round(max(all_z) * 1.25, 1)
        d["is3d"] = True
    return d


def _canvas_tab_html(si: int) -> str:
    return (
        f"<div class='canvas-wrap'>"
        f"<canvas id='drone-canvas-{si}' class='drone-canvas'></canvas>"
        f"<div class='canvas-ctrl'>"
        f"<button class='cbtn cbtn-reset' id='ctrl-reset-{si}'>&#9632;</button>"
        f"<button class='cbtn cbtn-play'  id='ctrl-play-{si}'>&#9654; Play</button>"
        f"<button class='cbtn cbtn-pause' id='ctrl-pause-{si}'>&#9646;&#9646; Pause</button>"
        f"<span class='speed-sep'>Speed:</span>"
        f"<button class='speed-btn' data-speed='0.5' data-si='{si}'>0.5×</button>"
        f"<button class='speed-btn speed-active' data-speed='1' data-si='{si}'>1×</button>"
        f"<button class='speed-btn' data-speed='2' data-si='{si}'>2×</button>"
        f"<button class='speed-btn' data-speed='4' data-si='{si}'>4×</button>"
        f"<span class='time-display' id='time-display-{si}'>t = 0.0 s</span>"
        f"</div>"
        f"<input type='range' id='time-slider-{si}' class='time-slider' min='0' max='1000' value='0'>"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Plotly chart helpers (unchanged)
# ---------------------------------------------------------------------------

def _make_2d_figure(primary, simulated, result, title):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=TEXT_COLOR), x=0.01),
        xaxis=dict(title="X (m)", showgrid=True, gridcolor=GRID_COLOR,
                   color=TEXT_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(title="Y (m)", showgrid=True, gridcolor=GRID_COLOR,
                   color=TEXT_COLOR, zerolinecolor=GRID_COLOR, scaleanchor="x"),
        plot_bgcolor=CHART_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=TEXT_COLOR, family="Inter, Segoe UI, Arial"),
        hovermode="closest",
        legend=dict(x=1.01, y=1, bgcolor="rgba(22,27,34,0.9)",
                    bordercolor="#30363d", borderwidth=1,
                    font=dict(color=TEXT_COLOR, size=12)),
        margin=dict(l=60, r=180, t=55, b=55), height=480,
    )
    _, pts_p = sample_trajectory(primary, n_points=300)
    fig.add_trace(go.Scatter(
        x=pts_p[:, 0].tolist(), y=pts_p[:, 1].tolist(), mode="lines",
        name=f"Primary ({primary.drone_id})",
        line=dict(color=PRIMARY_COLOR, width=3),
        hovertemplate="<b>Primary</b><br>x=%{x:.1f} m<br>y=%{y:.1f} m<extra></extra>",
    ))
    wpts_p = np.array([wp.to_array() for wp in primary.waypoints])
    fig.add_trace(go.Scatter(
        x=wpts_p[:, 0].tolist(), y=wpts_p[:, 1].tolist(),
        mode="markers+text",
        text=[f"W{i}" for i in range(len(primary.waypoints))],
        textposition="top right", textfont=dict(color=PRIMARY_COLOR, size=11),
        name=f"{primary.drone_id} waypoints",
        marker=dict(color=PRIMARY_COLOR, size=10, symbol="diamond"),
        showlegend=False,
    ))
    for k, sim in enumerate(simulated):
        color = SIM_COLORS[k % len(SIM_COLORS)]
        _, pts_s = sample_trajectory(sim, n_points=300)
        fig.add_trace(go.Scatter(
            x=pts_s[:, 0].tolist(), y=pts_s[:, 1].tolist(), mode="lines",
            name=f"Sim {sim.drone_id}",
            line=dict(color=color, width=2, dash="dash"),
            hovertemplate=f"<b>Sim {sim.drone_id}</b><br>x=%{{x:.1f}} m<br>y=%{{y:.1f}} m<extra></extra>",
        ))
        wpts_s = np.array([wp.to_array() for wp in sim.waypoints])
        fig.add_trace(go.Scatter(
            x=wpts_s[:, 0].tolist(), y=wpts_s[:, 1].tolist(), mode="markers",
            marker=dict(color=color, size=8, symbol="square"), showlegend=False,
        ))
    for c in result.conflicts:
        fig.add_trace(go.Scatter(
            x=[float(c.location[0])], y=[float(c.location[1])],
            mode="markers+text", name=f"Conflict: {c.conflicting_drone_id}",
            marker=dict(color=CONFLICT_COLOR, size=20, symbol="x",
                        line=dict(color=CONFLICT_COLOR, width=3)),
            text=[f"  t={c.time:.1f}s"], textposition="middle right",
            textfont=dict(color=CONFLICT_COLOR, size=11),
            hovertemplate=(
                f"<b>CONFLICT</b><br>Drone: {c.conflicting_drone_id}<br>"
                f"Time: {c.time:.2f} s<br>"
                f"({c.location[0]:.1f}, {c.location[1]:.1f})<br>"
                f"Sep: {c.separation:.2f} m<extra></extra>"
            ),
        ))
    return fig


def _make_distance_figure(primary, simulated, result, safety_buffer, title):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=TEXT_COLOR), x=0.01),
        xaxis=dict(title="Time (s)", showgrid=True, gridcolor=GRID_COLOR,
                   color=TEXT_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(title="Separation (m)", showgrid=True, gridcolor=GRID_COLOR,
                   color=TEXT_COLOR, zerolinecolor=GRID_COLOR),
        plot_bgcolor=CHART_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=TEXT_COLOR, family="Inter, Segoe UI, Arial"),
        hovermode="x unified",
        legend=dict(bgcolor="rgba(22,27,34,0.9)", bordercolor="#30363d",
                    borderwidth=1, font=dict(color=TEXT_COLOR, size=12)),
        margin=dict(l=60, r=180, t=55, b=55), height=360,
    )
    seg_times_p = compute_segment_times(primary)
    t0_p, t1_p = seg_times_p[0][0], seg_times_p[-1][1]
    any_overlap = False
    for k, sim in enumerate(simulated):
        color = SIM_COLORS[k % len(SIM_COLORS)]
        seg_s = compute_segment_times(sim)
        t0_s, t1_s = seg_s[0][0], seg_s[-1][1]
        t_lo, t_hi = max(t0_p, t0_s), min(t1_p, t1_s)
        if t_lo >= t_hi:
            fig.add_trace(go.Scatter(
                x=[t0_s, t1_s], y=[safety_buffer * 2, safety_buffer * 2],
                mode="lines+markers", name=f"Sim {sim.drone_id} (no overlap)",
                line=dict(color=color, width=6, dash="dot"),
                marker=dict(size=8, color=color),
                hovertemplate=(
                    f"<b>Sim {sim.drone_id}</b><br>"
                    f"Flies: {t0_s:.1f}s → {t1_s:.1f}s<br>"
                    f"Primary: {t0_p:.1f}s → {t1_p:.1f}s<br>"
                    "<i>No time overlap — safe</i><extra></extra>"
                ),
            ))
            continue
        any_overlap = True
        t_vals = np.linspace(t_lo, t_hi, 500)
        dists = []
        for t in t_vals:
            pp = position_at_time(primary, t)
            ps = position_at_time(sim, t)
            dists.append(float(np.linalg.norm(pp - ps)) if pp is not None and ps is not None else None)
        fig.add_trace(go.Scatter(
            x=t_vals.tolist(), y=dists, mode="lines",
            name=f"Sim {sim.drone_id}", line=dict(color=color, width=2),
            hovertemplate=f"<b>Sim {sim.drone_id}</b><br>t=%{{x:.2f}} s<br>dist=%{{y:.2f}} m<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=[t0_p, t1_p], y=[safety_buffer * 2.5, safety_buffer * 2.5],
        mode="lines+markers", name=f"Primary ({primary.drone_id}) window",
        line=dict(color=PRIMARY_COLOR, width=6), marker=dict(size=8, color=PRIMARY_COLOR),
        hovertemplate=f"<b>Primary</b><br>Flies: {t0_p:.1f}s → {t1_p:.1f}s<extra></extra>",
    ))
    fig.add_hline(y=safety_buffer, line=dict(color=CONFLICT_COLOR, dash="dash", width=1.5),
                  annotation_text=f"Safety buffer ({safety_buffer} m)",
                  annotation_position="top right",
                  annotation_font=dict(color=CONFLICT_COLOR, size=11))
    fig.add_hrect(y0=0, y1=safety_buffer, fillcolor=CONFLICT_COLOR, opacity=0.06, line_width=0)
    for c in result.conflicts:
        fig.add_vline(x=c.time, line=dict(color=CONFLICT_COLOR, dash="dot", width=1.5),
                      annotation_text=f"t={c.time:.1f}s",
                      annotation_font=dict(color=CONFLICT_COLOR, size=10),
                      annotation_position="top")
    if not any_overlap:
        fig.add_annotation(
            text="No temporal overlap — SAFE by design",
            xref="paper", yref="paper", x=0.5, y=0.35,
            showarrow=False, font=dict(color="#3fb950", size=13), align="center",
            bgcolor="rgba(63,185,80,0.08)", bordercolor="#3fb950",
            borderwidth=1, borderpad=10,
        )
    return fig


def _make_3d_figure(primary, simulated, result, title):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=TEXT_COLOR), x=0.01),
        paper_bgcolor=PAPER_BG,
        font=dict(color=TEXT_COLOR, family="Inter, Segoe UI, Arial"),
        scene=dict(
            xaxis=dict(title="X (m)", backgroundcolor=CHART_BG, gridcolor=GRID_COLOR, color=TEXT_COLOR),
            yaxis=dict(title="Y (m)", backgroundcolor=CHART_BG, gridcolor=GRID_COLOR, color=TEXT_COLOR),
            zaxis=dict(title="Alt Z (m)", backgroundcolor=CHART_BG, gridcolor=GRID_COLOR, color=TEXT_COLOR),
            bgcolor=CHART_BG,
        ),
        legend=dict(bgcolor="rgba(22,27,34,0.9)", bordercolor="#30363d",
                    borderwidth=1, font=dict(color=TEXT_COLOR, size=12)),
        margin=dict(l=0, r=0, t=55, b=0), height=500,
    )
    _, pts_p = sample_trajectory(primary, n_points=300)
    fig.add_trace(go.Scatter3d(
        x=pts_p[:, 0].tolist(), y=pts_p[:, 1].tolist(), z=pts_p[:, 2].tolist(),
        mode="lines", name=f"Primary ({primary.drone_id})",
        line=dict(color=PRIMARY_COLOR, width=6),
    ))
    for k, sim in enumerate(simulated):
        color = SIM_COLORS[k % len(SIM_COLORS)]
        _, pts_s = sample_trajectory(sim, n_points=300)
        fig.add_trace(go.Scatter3d(
            x=pts_s[:, 0].tolist(), y=pts_s[:, 1].tolist(), z=pts_s[:, 2].tolist(),
            mode="lines", name=f"Sim {sim.drone_id}", line=dict(color=color, width=4),
        ))
    for c in result.conflicts:
        fig.add_trace(go.Scatter3d(
            x=[float(c.location[0])], y=[float(c.location[1])], z=[float(c.location[2])],
            mode="markers", name=f"Conflict: {c.conflicting_drone_id}",
            marker=dict(color=CONFLICT_COLOR, size=12, symbol="x"),
            hovertemplate=(
                f"<b>CONFLICT</b><br>Drone: {c.conflicting_drone_id}<br>"
                f"t={c.time:.2f} s<br>"
                f"({c.location[0]:.1f}, {c.location[1]:.1f}, {c.location[2]:.1f})<br>"
                f"Sep: {c.separation:.2f} m<extra></extra>"
            ),
        ))
    return fig


def _make_gantt_figure(primary, simulated, title):
    import plotly.graph_objects as go
    fig = go.Figure()
    all_missions = [primary] + list(simulated)
    colors = [PRIMARY_COLOR] + [SIM_COLORS[k % len(SIM_COLORS)] for k in range(len(simulated))]
    labels = [f"Primary ({primary.drone_id})"] + [f"Sim {sim.drone_id}" for sim in simulated]
    for mission, color, label in zip(all_missions, colors, labels):
        seg = compute_segment_times(mission)
        t0, t1 = seg[0][0], seg[-1][1]
        fig.add_trace(go.Bar(
            y=[label], x=[t1 - t0], base=[t0], orientation="h",
            name=label, showlegend=False, width=0.5,
            marker=dict(color=color, opacity=0.75, line=dict(color=color, width=1.5)),
            hovertemplate=(
                f"<b>{label}</b><br>Start: {t0:.1f} s<br>"
                f"End: {t1:.1f} s<br>Duration: {t1-t0:.1f} s<extra></extra>"
            ),
        ))
    fig.update_layout(
        title=dict(text=f"{title} — Flight Timeline",
                   font=dict(size=14, color=TEXT_COLOR), x=0.01),
        xaxis=dict(title="Time (s)", showgrid=True, gridcolor=GRID_COLOR,
                   color=TEXT_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(color=TEXT_COLOR, showgrid=False, autorange="reversed"),
        plot_bgcolor=CHART_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=TEXT_COLOR, family="Inter, Segoe UI, Arial"),
        barmode="overlay", showlegend=False,
        height=max(180, 60 + len(all_missions) * 50),
        margin=dict(l=140, r=40, t=50, b=50), hovermode="closest",
    )
    return fig


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_html_report(
    scenarios: list,
    safety_buffer: float = 8.0,
    save_path: str = "outputs/report.html",
) -> str:
    import plotly.io as pio

    total     = len(scenarios)
    n_clear   = sum(1 for s in scenarios if s["result"].is_clear)
    n_conflict = total - n_clear
    total_conf = sum(len(s["result"].conflicts) for s in scenarios)
    buf_vals   = sorted(set(s.get("buffer", safety_buffer) for s in scenarios))
    buf_disp   = f"{buf_vals[0]} m" if len(buf_vals) == 1 else f"{buf_vals[0]}–{buf_vals[-1]} m"

    stats_html = f"""
    <div class='stats-grid'>
      <div class='stat-card'><div class='stat-value'>{total}</div><div class='stat-label'>Total Scenarios</div></div>
      <div class='stat-card stat-card-clear'><div class='stat-value'>{n_clear}</div><div class='stat-label'>Missions Clear</div></div>
      <div class='stat-card stat-card-conflict'><div class='stat-value'>{n_conflict}</div><div class='stat-label'>Conflicts Found</div></div>
      <div class='stat-card stat-card-buf'><div class='stat-value'>{buf_disp}</div><div class='stat-label'>Safety Buffer</div></div>
    </div>"""

    nav_html = "".join(
        f"<a class='nav-link' href='#scenario-{i}'>{s['name'].split('—')[0].strip()}</a>"
        for i, s in enumerate(scenarios)
    )

    # Pre-build canvas data for all scenarios
    all_canvas_data = {}
    for i, s in enumerate(scenarios):
        all_canvas_data[i] = _build_canvas_data(
            s["primary"], s["simulated"], s["result"],
            is_3d=s.get("is_3d", False),
            safety_buffer=s.get("buffer", safety_buffer),
        )

    blocks = []
    summary_rows = ""

    for i, s in enumerate(scenarios):
        name      = s["name"]
        primary   = s["primary"]
        simulated = s["simulated"]
        result    = s["result"]
        buf       = s.get("buffer", safety_buffer)
        is_3d     = s.get("is_3d", False)

        badge = ("<span class='badge badge-clear'>&#10003; CLEAR</span>"
                 if result.is_clear else
                 "<span class='badge badge-conflict'>&#10007; CONFLICT DETECTED</span>")
        feasible_html = ("<span class='pill pill-ok'>&#10003; Feasible</span>"
                         if result.feasible else
                         "<span class='pill pill-err'>&#10007; Infeasible</span>")

        if result.conflicts:
            rows = "".join(
                f"<tr><td><strong>{c.conflicting_drone_id}</strong></td>"
                f"<td>{c.time:.2f} s</td>"
                f"<td>({c.location[0]:.1f}, {c.location[1]:.1f}, {c.location[2]:.1f})</td>"
                f"<td class='sep-danger'>{c.separation:.2f} m</td>"
                f"<td>{c.safety_buffer:.1f} m</td></tr>"
                for c in result.conflicts
            )
            conflict_sec = (
                f"<div class='conflict-box'>"
                f"<div class='conflict-box-title'>&#9888; {len(result.conflicts)} Conflict(s) Detected</div>"
                f"<table class='ctable'><thead><tr>"
                f"<th>Drone ID</th><th>Time</th><th>Location (x,y,z)</th>"
                f"<th>Separation</th><th>Buffer</th>"
                f"</tr></thead><tbody>{rows}</tbody></table></div>"
            )
        else:
            conflict_sec = "<div class='clear-box'>&#10003; No conflicts — mission is safe.</div>"

        # Gantt
        fig_gantt = _make_gantt_figure(primary, simulated, name)
        gantt_html = pio.to_html(fig_gantt, full_html=False, include_plotlyjs=False)

        # Path (2D or 3D)
        if is_3d:
            fig_path = _make_3d_figure(primary, simulated, result, f"{name} — 3D Paths")
        else:
            fig_path = _make_2d_figure(primary, simulated, result, f"{name} — Paths")
        path_html = pio.to_html(fig_path, full_html=False, include_plotlyjs=False)

        # Distance
        fig_dist = _make_distance_figure(primary, simulated, result, buf, f"{name} — Separation vs Time")
        dist_html = pio.to_html(fig_dist, full_html=False, include_plotlyjs=False)

        # Tabs
        canvas_html = _canvas_tab_html(i)
        if not is_3d:
            tabs = (
                f"<div class='tab-bar'>"
                f"<button class='tab-btn tab-active' onclick='showTab(this,\"canvas-{i}\")'>&#128640; Live Drone View</button>"
                f"<button class='tab-btn' onclick='showTab(this,\"path-{i}\")'>&#128205; Paths</button>"
                f"<button class='tab-btn' onclick='showTab(this,\"dist-{i}\")'>&#128202; Separation</button>"
                f"<button class='tab-btn' onclick='showTab(this,\"gantt-{i}\")'>&#9200; Timeline</button>"
                f"</div>"
                f"<div id='canvas-{i}' class='tab-content tab-active'>{canvas_html}</div>"
                f"<div id='path-{i}' class='tab-content'>{path_html}</div>"
                f"<div id='dist-{i}' class='tab-content'>{dist_html}</div>"
                f"<div id='gantt-{i}' class='tab-content'>{gantt_html}</div>"
            )
        else:
            tabs = (
                f"<div class='tab-bar'>"
                f"<button class='tab-btn tab-active' onclick='showTab(this,\"canvas-{i}\")'>&#128640; Live 3D View</button>"
                f"<button class='tab-btn' onclick='showTab(this,\"path-{i}\")'>&#127757; 3D Paths</button>"
                f"<button class='tab-btn' onclick='showTab(this,\"dist-{i}\")'>&#128202; Separation</button>"
                f"<button class='tab-btn' onclick='showTab(this,\"gantt-{i}\")'>&#9200; Timeline</button>"
                f"</div>"
                f"<div id='canvas-{i}' class='tab-content tab-active'>{canvas_html}</div>"
                f"<div id='path-{i}' class='tab-content'>{path_html}</div>"
                f"<div id='dist-{i}' class='tab-content'>{dist_html}</div>"
                f"<div id='gantt-{i}' class='tab-content'>{gantt_html}</div>"
            )

        blocks.append(
            f"<div class='card' id='scenario-{i}'>"
            f"<div class='card-header' onclick='toggleCard(this)'>"
            f"<span class='card-title'>{name}</span>"
            f"<div style='display:flex;align-items:center;gap:10px;'>{badge}"
            f"<span class='collapse-icon'>&#9650;</span></div></div>"
            f"<div class='card-body'>{feasible_html}{conflict_sec}{tabs}</div>"
            f"</div>"
        )

        st_cell = (f"<td><span class='badge badge-clear sm'>&#10003; CLEAR</span></td>"
                   if result.is_clear else
                   f"<td><span class='badge badge-conflict sm'>&#10007; CONFLICT</span></td>")
        feas_cell = (f"<td class='ok-text'>&#10003; Yes</td>" if result.feasible
                     else f"<td class='err-text'>&#10007; No</td>")
        summary_rows += (
            f"<tr><td><a class='tbl-link' href='#scenario-{i}'>{name}</a></td>"
            f"{st_cell}<td class='center'>{len(result.conflicts)}</td>{feas_cell}</tr>"
        )

    drone_data_json = json.dumps(all_canvas_data, separators=(",", ":"))

    # Build HTML — JS split to avoid f-string brace escaping for canvas JS
    html_part1 = f"""<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='UTF-8'>
  <meta name='viewport' content='width=device-width,initial-scale=1'>
  <title>UAV Deconfliction Report &mdash; FlytBase</title>
  <script src='https://cdn.plot.ly/plotly-2.27.0.min.js'></script>
  <script>window.droneData = {drone_data_json};</script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;}}
    body{{font-family:'Inter','Segoe UI',Arial,sans-serif;background:#0d1117;color:#e6edf3;min-height:100vh;padding:0 0 60px;}}
    .topbar{{background:#161b22;border-bottom:1px solid #21262d;padding:0 48px;display:flex;align-items:center;position:sticky;top:0;z-index:100;height:52px;}}
    .topbar-logo{{font-size:17px;font-weight:700;color:#00d4ff;letter-spacing:1px;white-space:nowrap;padding-right:16px;margin-right:16px;border-right:1px solid #30363d;}}
    .topbar-sub{{font-size:13px;color:#8b949e;white-space:nowrap;margin-right:24px;}}
    .nav-links{{display:flex;gap:4px;overflow-x:auto;flex:1;}}
    .nav-link{{font-size:12px;color:#8b949e;text-decoration:none;padding:6px 12px;border-radius:6px;white-space:nowrap;transition:background .15s,color .15s;}}
    .nav-link:hover{{background:#21262d;color:#e6edf3;}}
    .wrapper{{max-width:1200px;margin:0 auto;padding:36px 32px;}}
    .page-heading{{margin-bottom:28px;}}
    .page-heading h1{{font-size:26px;font-weight:700;color:#e6edf3;margin-bottom:6px;}}
    .page-heading p{{font-size:13px;color:#8b949e;line-height:1.7;}}
    .page-heading p span{{color:#00d4ff;font-weight:500;}}
    .stats-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:28px;}}
    .stat-card{{background:#161b22;border:1px solid #21262d;border-radius:10px;padding:20px 18px;text-align:center;}}
    .stat-card-clear{{border-color:rgba(63,185,80,.35);}}
    .stat-card-conflict{{border-color:rgba(248,81,73,.35);}}
    .stat-card-buf{{border-color:rgba(0,212,255,.35);}}
    .stat-value{{font-size:32px;font-weight:700;color:#e6edf3;line-height:1.1;margin-bottom:6px;}}
    .stat-card-clear .stat-value{{color:#3fb950;}}
    .stat-card-conflict .stat-value{{color:#f85149;}}
    .stat-card-buf .stat-value{{color:#00d4ff;font-size:22px;}}
    .stat-label{{font-size:12px;color:#8b949e;font-weight:500;text-transform:uppercase;letter-spacing:.6px;}}
    .summary-card{{background:#161b22;border:1px solid #21262d;border-radius:10px;overflow:hidden;margin-bottom:36px;}}
    .summary-card-title{{padding:12px 20px;font-size:12px;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:.8px;border-bottom:1px solid #21262d;}}
    .summary-table{{width:100%;border-collapse:collapse;font-size:14px;}}
    .summary-table th{{background:#0d1117;color:#8b949e;padding:10px 18px;text-align:left;font-weight:500;font-size:12px;text-transform:uppercase;letter-spacing:.6px;border-bottom:1px solid #21262d;}}
    .summary-table td{{padding:12px 18px;border-bottom:1px solid #21262d;color:#e6edf3;}}
    .summary-table tr:last-child td{{border-bottom:none;}}
    .summary-table tr:hover td{{background:#1c2128;}}
    .center{{text-align:center;}} .ok-text{{color:#3fb950;font-weight:600;}} .err-text{{color:#f85149;font-weight:600;}}
    .tbl-link{{color:#e6edf3;text-decoration:none;}} .tbl-link:hover{{color:#00d4ff;text-decoration:underline;}}
    .badge{{display:inline-flex;align-items:center;gap:5px;padding:5px 12px;border-radius:20px;font-size:12px;font-weight:700;letter-spacing:.5px;white-space:nowrap;}}
    .badge.sm{{padding:3px 10px;font-size:11px;}}
    .badge-clear{{background:rgba(63,185,80,.15);color:#3fb950;border:1px solid #3fb950;}}
    .badge-conflict{{background:rgba(248,81,73,.15);color:#f85149;border:1px solid #f85149;}}
    .card{{background:#161b22;border:1px solid #21262d;border-radius:10px;margin-bottom:28px;overflow:hidden;scroll-margin-top:64px;}}
    .card-header{{display:flex;align-items:center;justify-content:space-between;padding:15px 22px;border-bottom:1px solid #21262d;background:#1c2128;flex-wrap:wrap;gap:10px;cursor:pointer;user-select:none;transition:background .15s;}}
    .card-header:hover{{background:#22272e;}}
    .card-title{{font-size:15px;font-weight:600;color:#e6edf3;}}
    .collapse-icon{{font-size:12px;color:#8b949e;}}
    .card-body{{padding:20px 22px 24px;}}
    .pill{{display:inline-block;padding:5px 12px;border-radius:6px;font-size:12px;font-weight:500;margin-bottom:14px;}}
    .pill-ok{{background:rgba(63,185,80,.1);color:#3fb950;border:1px solid rgba(63,185,80,.3);}}
    .pill-err{{background:rgba(248,81,73,.1);color:#f85149;border:1px solid rgba(248,81,73,.3);}}
    .conflict-box{{background:rgba(248,81,73,.07);border:1px solid rgba(248,81,73,.3);border-radius:8px;padding:14px 16px;margin-bottom:16px;}}
    .conflict-box-title{{font-size:13px;font-weight:600;color:#f85149;margin-bottom:12px;}}
    .ctable{{width:100%;border-collapse:collapse;font-size:13px;}}
    .ctable th{{background:rgba(248,81,73,.12);color:#f85149;padding:7px 12px;text-align:left;font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.5px;}}
    .ctable td{{padding:8px 12px;border-top:1px solid rgba(248,81,73,.1);color:#e6edf3;font-family:'Courier New',monospace;font-size:12px;}}
    .sep-danger{{color:#f85149;font-weight:700;}}
    .clear-box{{background:rgba(63,185,80,.07);border:1px solid rgba(63,185,80,.3);border-radius:8px;padding:10px 16px;font-size:13px;color:#3fb950;margin-bottom:16px;}}
    .tab-bar{{display:flex;gap:4px;margin-bottom:12px;border-bottom:1px solid #21262d;padding-bottom:8px;}}
    .tab-btn{{background:none;border:1px solid transparent;color:#8b949e;font-family:'Inter',sans-serif;font-size:13px;font-weight:500;padding:6px 14px;border-radius:6px;cursor:pointer;transition:all .15s;white-space:nowrap;}}
    .tab-btn:hover{{background:#21262d;color:#e6edf3;}}
    .tab-btn.tab-active{{background:rgba(0,212,255,.1);border-color:rgba(0,212,255,.4);color:#00d4ff;}}
    .tab-content{{display:none;}} .tab-content.tab-active{{display:block;}}
    .canvas-wrap{{position:relative;background:#0d1117;border-radius:8px;overflow:hidden;}}
    .drone-canvas{{display:block;width:100%;background:#0d1117;}}
    .canvas-ctrl{{display:flex;align-items:center;gap:8px;padding:10px 14px;background:#161b22;border-top:1px solid #21262d;flex-wrap:wrap;}}
    .cbtn{{background:#21262d;border:1px solid #30363d;color:#e6edf3;font-size:12px;font-weight:600;padding:6px 14px;border-radius:6px;cursor:pointer;transition:background .15s;}}
    .cbtn:hover{{background:#2d333b;}}
    .cbtn-play{{background:rgba(63,185,80,.15);border-color:#3fb950;color:#3fb950;}}
    .cbtn-play:hover{{background:rgba(63,185,80,.25);}}
    .cbtn-pause{{background:rgba(255,193,7,.1);border-color:#ffc107;color:#ffc107;}}
    .cbtn-pause:hover{{background:rgba(255,193,7,.2);}}
    .cbtn-reset{{padding:6px 10px;}}
    .speed-sep{{font-size:12px;color:#8b949e;margin-left:4px;}}
    .speed-btn{{background:#21262d;border:1px solid #30363d;color:#8b949e;font-size:11px;font-weight:600;padding:4px 10px;border-radius:5px;cursor:pointer;transition:all .15s;}}
    .speed-btn:hover{{color:#e6edf3;border-color:#8b949e;}}
    .speed-btn.speed-active{{background:rgba(0,212,255,.12);border-color:#00d4ff;color:#00d4ff;}}
    .time-display{{margin-left:auto;font-family:'Courier New',monospace;font-size:13px;color:#00d4ff;font-weight:600;padding:4px 10px;background:rgba(0,212,255,.07);border:1px solid rgba(0,212,255,.2);border-radius:6px;}}
    .time-slider{{width:100%;margin:0;height:4px;-webkit-appearance:none;background:#21262d;outline:none;cursor:pointer;}}
    .time-slider::-webkit-slider-thumb{{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:#00d4ff;cursor:pointer;}}
    .time-slider::-moz-range-thumb{{width:14px;height:14px;border-radius:50%;background:#00d4ff;cursor:pointer;border:none;}}
    footer{{text-align:center;color:#484f58;font-size:12px;margin-top:16px;padding-top:16px;border-top:1px solid #21262d;}}
    @media(max-width:700px){{.stats-grid{{grid-template-columns:repeat(2,1fr);}} .topbar{{padding:0 16px;}} .wrapper{{padding:20px 16px;}}}}
  </style>
</head>
<body>
  <nav class='topbar'>
    <span class='topbar-logo'>FlytBase</span>
    <span class='topbar-sub'>UAV Strategic Deconfliction</span>
    <div class='nav-links'>{nav_html}</div>
  </nav>
  <div class='wrapper'>
    <div class='page-heading'>
      <h1>Deconfliction Analysis Report</h1>
      <p>
        <span>&#128640; Play</span> to watch drones fly with real quad-copter animation &nbsp;&bull;&nbsp;
        <span>Drag</span> time slider to scrub &nbsp;&bull;&nbsp;
        <span>Tabs</span> for paths / separation / timeline &nbsp;&bull;&nbsp;
        <span>Drag</span> 3D charts to rotate
      </p>
    </div>
    {stats_html}
    <div class='summary-card'>
      <div class='summary-card-title'>Mission Summary</div>
      <table class='summary-table'>
        <thead><tr><th>Scenario</th><th>Status</th><th>Conflicts</th><th>Feasible</th></tr></thead>
        <tbody>{summary_rows}</tbody>
      </table>
    </div>
    {chr(10).join(blocks)}
    <footer>Generated by UAV Deconfliction System &mdash; FlytBase Technical Assignment</footer>
  </div>
  <script>
    function showTab(btn,tabId){{
      var bar=btn.parentElement;
      bar.querySelectorAll('.tab-btn').forEach(function(b){{b.classList.remove('tab-active');}});
      btn.classList.add('tab-active');
      var body=bar.parentElement;
      body.querySelectorAll('.tab-content').forEach(function(c){{c.classList.remove('tab-active');}});
      var tgt=document.getElementById(tabId);
      tgt.classList.add('tab-active');
      if(tabId.startsWith('canvas-')){{
        var si=parseInt(tabId.replace('canvas-',''));
        setTimeout(function(){{if(window.__reinitCanvas) window.__reinitCanvas(si);}},30);
      }}
      setTimeout(function(){{
        tgt.querySelectorAll('.js-plotly-plot').forEach(function(p){{Plotly.relayout(p,{{autosize:true}});}});
      }},30);
    }}
    function toggleCard(hdr){{
      var body=hdr.nextElementSibling, icon=hdr.querySelector('.collapse-icon');
      var hidden=body.style.display==='none';
      body.style.display=hidden?'':'none';
      icon.innerHTML=hidden?'&#9650;':'&#9660;';
      if(hidden){{
        setTimeout(function(){{
          body.querySelectorAll('.tab-active .js-plotly-plot').forEach(function(p){{Plotly.relayout(p,{{autosize:true}});}});
          body.querySelectorAll('.tab-active .drone-canvas').forEach(function(c){{
            var si=parseInt(c.id.replace('drone-canvas-',''));
            if(window.__reinitCanvas) window.__reinitCanvas(si);
          }});
        }},50);
      }}
    }}
    document.querySelectorAll('.nav-link').forEach(function(l){{
      l.addEventListener('click',function(e){{
        e.preventDefault();
        var t=document.querySelector(this.getAttribute('href'));
        if(t) t.scrollIntoView({{behavior:'smooth',block:'start'}});
      }});
    }});
  </script>
  <script>
"""

    html_part2 = """
  </script>
</body>
</html>"""

    html = html_part1 + _CANVAS_JS + _CANVAS_3D_JS + html_part2

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html)
    return save_path

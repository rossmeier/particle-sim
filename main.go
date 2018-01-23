package main

import (
	"fmt"
	"log"
	"math"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/go-gl/mathgl/mgl32"

	"github.com/go-gl/gl/v4.1-core/gl"
	"github.com/go-gl/glfw/v3.2/glfw"
)

func init() {
	// This is needed to arrange that main() runs on main thread.
	// See documentation for functions that are only allowed to be called from the main thread.
	runtime.LockOSThread()
}

type forceFunc func(x, v mgl32.Vec2) mgl32.Vec2

func main() {
	err := glfw.Init()
	if err != nil {
		panic(err)
	}
	defer glfw.Terminate()

	// configure window options
	glfw.WindowHint(glfw.Samples, 4)
	//glfw.WindowHint(glfw.ContextVersionMajor, 4)
	//glfw.WindowHint(glfw.ContextVersionMinor, 1)

	// create window
	window, err := glfw.CreateWindow(1280, 720, "Testing", nil, nil)
	if err != nil {
		panic(err)
	}

	// make window context available for gl operations
	window.MakeContextCurrent()

	// configure window key input mode
	window.SetInputMode(glfw.StickyKeysMode, glfw.True)

	// init opengl
	err = gl.Init()
	if err != nil {
		log.Fatal(err)
	}

	window.SetSizeCallback(func(w *glfw.Window, width int, height int) {
		gl.Viewport(0, 0, int32(width), int32(height))
	})

	// compile shaders
	program, err := newProgram(vertexshader, fragmentshader)
	if err != nil {
		log.Fatal(err)
	}

	// create vertex buffer
	var vertexArrayID uint32
	gl.GenVertexArrays(1, &vertexArrayID)
	gl.BindVertexArray(vertexArrayID)

	// init data for physics
	points := make([]mgl32.Vec2, 100000)
	for i := range points {
		phi := 2 * math.Pi / float64(len(points)) * float64(i)
		points[i][0] = float32(math.Cos(phi)) * .5
		points[i][1] = float32(math.Sin(phi)) * .5
	}
	velocities := make([]mgl32.Vec2, len(points))
	forceFuncs := make(chan forceFunc, 1)

	// init slice for vertex data
	gVertexBufferData := make([]float32, len(points)*24)
	var gVertexBufferDataMutex sync.Mutex

	// start physics
	asyncPhysics(points, velocities, forceFuncs, gVertexBufferData, gVertexBufferDataMutex)

	// init vertex buffer in gpu
	var vertexbuffer uint32
	gl.GenBuffers(1, &vertexbuffer)
	gl.BindBuffer(gl.ARRAY_BUFFER, vertexbuffer)

	lastupdate := time.Now()
	for !window.ShouldClose() {
		// record timing
		diff := float32(time.Since(lastupdate)) / float32(time.Second)
		lastupdate = time.Now()
		window.SetTitle(fmt.Sprintf("Testing (%f FPS)", 1/diff))

		// handle physics
		x, y := window.GetCursorPos()
		w, h := window.GetSize()
		glX, glY := mgl32.ScreenToGLCoords(int(x), int(y), w, h)
		glPos := mgl32.Vec2{glX, glY}

		force := float32(0)
		if window.GetMouseButton(glfw.MouseButtonLeft) == glfw.Press {
			force = 1
		} else if window.GetMouseButton(glfw.MouseButtonRight) == glfw.Press {
			force = -1
		} else if window.GetMouseButton(glfw.MouseButtonMiddle) == glfw.Press {
			force = 3
		}
		select {
		case forceFuncs <- func(x, v mgl32.Vec2) mgl32.Vec2 {
			r := glPos.Sub(x).Len()
			return glPos.Sub(x).Normalize().Mul(0.005 * force / ((r) + .01)).Add(v.Mul(-v.Len() * 300))
		}:
		default:
		}

		// reupdate buffer
		gVertexBufferDataMutex.Lock()
		gl.BufferData(gl.ARRAY_BUFFER, len(gVertexBufferData)*4, gl.Ptr(gVertexBufferData), gl.STATIC_DRAW)
		gVertexBufferDataMutex.Unlock()

		// clear canvas
		gl.ClearColor(0.1, 0, .1, 0)
		gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

		// compile shader
		gl.UseProgram(program)

		// draw data from buffer
		gl.EnableVertexAttribArray(0)
		gl.BindBuffer(gl.ARRAY_BUFFER, vertexbuffer)
		gl.VertexAttribPointer(0, 4, gl.FLOAT, false, 0, nil)
		gl.DrawArrays(gl.TRIANGLES, 0, int32(len(gVertexBufferData)))
		gl.DisableVertexAttribArray(0)

		// show new image
		window.SwapBuffers()

		// handle window events
		glfw.PollEvents()
		if window.GetKey(glfw.KeyEscape) == glfw.Press {
			window.SetShouldClose(true)
		}
	}
}

var vertexshader = `
#version 330 core
#extension GL_ARB_enhanced_layouts : enable
layout(location=0) in vec3 pos;
layout(location=0,component=3) in float color;
out float c;

void main() {
	gl_Position.xyz = pos;
	gl_Position.w = 1.0;
	c = color;
}
` + "\x00"

var fragmentshader = `
#version 330 core
out vec3 color;
in float c;

void main() {
	color = vec3(1-c,c,0);
}
` + "\x00"

func asyncPhysics(x, v []mgl32.Vec2, force chan forceFunc, data []float32, dataMutex sync.Mutex) {
	t := time.Now()
	go func() {
		f := <-force
		for {
			dt := float32(time.Since(t)) / float32(time.Second)
			fmt.Println("Physics!", (1 / dt))
			t = time.Now()
			physics(dt, x, v, f)
			dataMutex.Lock()
			createVertexBuffer(x, v, data)
			dataMutex.Unlock()
			select {
			case f = <-force:
			default:
			}
		}
	}()
}

func physics(t float32, x, v []mgl32.Vec2, force forceFunc) {
	if len(x) > len(v) {
		panic("not enough velocities")
	}
	c := 0
	var wg sync.WaitGroup
	for i := 0; i < runtime.NumCPU(); i++ {
		vOld := v[i]
		v[i] = v[i].Add(force(x[i], v[i]).Mul(t))
		x[i] = x[i].Add(vOld.Add(v[i]))
		if x[i][0] < -1 {
			x[i][0] = -2 - x[i][0]
			v[i][0] = -v[i][0]
		}
		if x[i][0] > 1 {
			x[i][0] = 2 - x[i][0]
			v[i][0] = -v[i][0]
		}
		if x[i][1] < -1 {
			x[i][1] = -2 - x[i][1]
			v[i][1] = -v[i][1]
		}
		if x[i][1] > 1 {
			x[i][1] = 2 - x[i][1]
			v[i][1] = -v[i][1]
		}
	}
}

func createVertexBuffer(in []mgl32.Vec2, v []mgl32.Vec2, out []float32) {
	if len(out) < 24*len(in) {
		panic("out to short")
	}
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < len(in); i++ {
			c := mgl32.Clamp(v[i].Len()*100, 0, 1)
			for j := 0; j < 6; j++ {
				out[24*i+4*j+2] = 0 // z-coordinate
				out[24*i+4*j+3] = c // color
			}
		}
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		d := float32(.001)
		for i := range in {
			// 1st triangle
			out[24*i+0] = -d + in[i][0] // x of 1st point
			out[24*i+1] = d + in[i][1]  // y of 1nd point

			out[24*i+4*1+0] = -d + in[i][0] // 2nd point
			out[24*i+4*1+1] = -d + in[i][1]

			out[24*i+4*2+0] = d + in[i][0] // 3rd point
			out[24*i+4*2+1] = -d + in[i][1]

			// 2nd triangle -> counterpart to form a square
			out[24*i+4*3+0] = d + in[i][0]
			out[24*i+4*3+1] = -d + in[i][1]

			out[24*i+4*4+0] = d + in[i][0]
			out[24*i+4*4+1] = d + in[i][1]

			out[24*i+4*5+0] = -d + in[i][0]
			out[24*i+4*5+1] = d + in[i][1]
		}
	}()
	wg.Wait()
}

func newProgram(vertexShaderSource, fragmentShaderSource string) (uint32, error) {
	vertexShader, err := compileShader(vertexShaderSource, gl.VERTEX_SHADER)
	if err != nil {
		return 0, err
	}

	fragmentShader, err := compileShader(fragmentShaderSource, gl.FRAGMENT_SHADER)
	if err != nil {
		return 0, err
	}

	program := gl.CreateProgram()

	gl.AttachShader(program, vertexShader)
	gl.AttachShader(program, fragmentShader)
	gl.LinkProgram(program)

	var status int32
	gl.GetProgramiv(program, gl.LINK_STATUS, &status)
	if status == gl.FALSE {
		var logLength int32
		gl.GetProgramiv(program, gl.INFO_LOG_LENGTH, &logLength)

		log := strings.Repeat("\x00", int(logLength+1))
		gl.GetProgramInfoLog(program, logLength, nil, gl.Str(log))

		return 0, fmt.Errorf("failed to link program: %v", log)
	}

	gl.DeleteShader(vertexShader)
	gl.DeleteShader(fragmentShader)

	return program, nil
}

func compileShader(source string, shaderType uint32) (uint32, error) {
	shader := gl.CreateShader(shaderType)

	csources, free := gl.Strs(source)
	gl.ShaderSource(shader, 1, csources, nil)
	free()
	gl.CompileShader(shader)

	var status int32
	gl.GetShaderiv(shader, gl.COMPILE_STATUS, &status)
	if status == gl.FALSE {
		var logLength int32
		gl.GetShaderiv(shader, gl.INFO_LOG_LENGTH, &logLength)

		log := strings.Repeat("\x00", int(logLength+1))
		gl.GetShaderInfoLog(shader, logLength, nil, gl.Str(log))

		return 0, fmt.Errorf("failed to compile %v: %v", source, log)
	}

	return shader, nil
}

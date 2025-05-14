using Newtonsoft.Json;
using System;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Unity.VisualScripting;
using UnityEditor;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.InputSystem;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
public class Game : MonoBehaviour
{ 
    public float                DurationBetweenUpdates;
    public RobotHeadSensors     HeadSensors;
    public BipedalRobotLimbs    Actuators;
    public Target               Target;
    public Slider               GameSpeedSlider;
    public float                EpisodeDuration = 10;
    public static float         SimulationSpeed = 3.5f;
    float startTime;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        SetupRendering();
        this.agent = AgentEndpoint.Singleton;
        this.startTime = Time.time;
        this.GameSpeedSlider.minValue = 0.5f;
        this.GameSpeedSlider.maxValue = 15f ;
        this.GameSpeedSlider.value = SimulationSpeed;

        StartPreparingScene();

        stepCount = 0;
    }
    static void SetupRendering(){
        Screen.SetResolution(640, 640, FullScreenMode.Windowed, new RefreshRate(){numerator = 30, denominator = 1});
    }
    static void StartPreparingScene(){
        ;
    }
    void Restart(){
        SceneManager.LoadScene(SceneManager.GetActiveScene().buildIndex);
    }
    
    AgentEndpoint       agent;
    RewardCalculator    rewardCalculator;
    int                 stepCount = 0;
    bool PlayerUpdate()
    {
        var timeScale = Time.timeScale;
        Time.timeScale = 0.01f;
        try
        {
            GameState CollectState() =>
                new GameState {
                    SensorsReading = this.HeadSensors.GetReading(),
                    IsFinished = this.Target.IsTouchingRobot(),
                    LimbsReading = this.Actuators.GetLimbReadings(),
                    
                };

            float CalculateReward(GameState state)
                => this.rewardCalculator.CalculateReward(state);

            void PrepareGame(GameState state)
            {
                this.rewardCalculator = new RewardCalculator(state);
                this.agent.InformGameHasStarted();
            }

            GameAction TalkToAgentAboutStep(GameState state, float reward)
            {
                return this.agent.SendStateAndRecvAction(state, reward);
                //return new GameAction {
                //    LimbsActivations = new LimbsActivation {
                //        LeftLimbAct = new LimbActivation { ShinInput = 0, ShoulderInput = 0, ThighInput = 0 },
                //        RightLimbAct = new LimbActivation { ShinInput = 0, ShoulderInput = 0, ThighInput = 0 }
                //    }
                //};
            }

            void ApplyAction(GameAction a)
            {
                this.Actuators.ApplyActivations(a.LimbsActivation);
            }


            if (stepCount == 0)
            {
                this.agent.Connect();
            }

            Debug.Log($"Step {stepCount}");
            var state = CollectState();

            if(stepCount == 0)
            {
                PrepareGame(state);
            }

            Debug.Log($"State: {state}");
            var reward = CalculateReward(state);
            Debug.Log($"Reward: {reward}");
            var action = TalkToAgentAboutStep(state, reward);
            Debug.Log($"Action: {action}");

            ApplyAction(action);
            stepCount += 1;
        }
        catch
        {
            throw;
        }
        finally
        {
            Time.timeScale = timeScale;
        }
        return true;
    }

    float   nextUpdate;
    bool    isRunning;
    void FixedUpdate()
    {
        var now = Time.fixedTime;
        SimulationSpeed = this.GameSpeedSlider.value;
        Time.timeScale = SimulationSpeed;
        if (now >= nextUpdate)
        {
            PlayerUpdate();
            nextUpdate = now + DurationBetweenUpdates;
        }
        if (Time.time -  startTime > EpisodeDuration) 
        {
            this.Restart();
        } 
    }
}

[Serializable]
public struct GameAction
{
    public BipedalLimbsActivation LimbsActivation;
    public override string ToString() =>
$@"GameAction{{
    LimbsActivation: {this.LimbsActivation.ToString().Replace("\n", "\n\t")}
}}";
}

[Serializable]
public struct GameState
{
    public bool                 IsFinished;
    public SensorsReading   SensorsReading;
    public BipedalLimbsReading  LimbsReading;
    public override string ToString() =>
$@"GameState{{
    IsFinished: {IsFinished},
    HeadSensorsReading: {SensorsReading.ToString().Replace("\n", "\n\t")},
    LimbsReading: {LimbsReading.ToString().Replace("\n", "\n\t")}
}}";
     
}

[Serializable]
public struct GameStateAndReward
{
    public float Reward;
    public GameState State;
}
public class RewardCalculator
{
    float previousTime = 0.0f; 
    public float RewardForNotStanding = -100.0f;
    public float RewardForHittingHead = -1000.0f;
    public float RewardPerMeterCloserToTarget = 100.0f;
    public float CorrectDistanceToFloor = 0.60f;
    public float MaxDistanceToFloorDiff = 0.65f;

    float previousDistanceToTarget;
    public RewardCalculator(GameState firstState)
    {
        previousTime = Time.time;
        previousDistanceToTarget = ((Vector3)firstState.SensorsReading.TargetPos).magnitude;
    }
    public float CalculateReward(GameState gameState)
    {
        var now = Time.time;
        var rewardDeltaTime = now - previousTime;
        //var uprightError = 1 - gameState.HeadSensorsReading.UpOrientation.y;
        var distanceFromFloorError = Mathf.Pow(Mathf.Min((Mathf.Abs(gameState.SensorsReading.FloorDist - this.CorrectDistanceToFloor) / MaxDistanceToFloorDiff), 1.0f), 2);
        var notStandingPunishment = (distanceFromFloorError) * RewardForNotStanding * rewardDeltaTime;

        var currentDistanceToTarget = ((Vector3)gameState.SensorsReading.TargetPos).magnitude;
        var deltaDistance = - currentDistanceToTarget + previousDistanceToTarget;
        var rewardForWalkingTorwardsTarget = deltaDistance * RewardPerMeterCloserToTarget;

        var hittingHeadPunishment = (gameState.SensorsReading.FloorDist < 0.35) ? RewardForHittingHead  * rewardDeltaTime : 0  ;

        previousDistanceToTarget = currentDistanceToTarget;
        previousTime = now;

        return notStandingPunishment + rewardForWalkingTorwardsTarget + hittingHeadPunishment;
    }
}

public class AgentEndpoint
{
    public static AgentEndpoint Singleton = new AgentEndpoint(); 

    Socket socket; 
    private AgentEndpoint() {
        this.socket = new Socket(SocketType.Stream, ProtocolType.Tcp);
    }
    public void Connect()
    {
        if (!this.socket.Connected)
        {
            this.socket.Connect(IPAddress.Loopback, 8080);
        }
    }
    public void InformGameHasStarted()
    {
        this.Send(UTF8Encoding.UTF8.GetBytes("GAME STARTED"));
    }
    void Send(byte[] bytes)
    {
        byte[] getLenBytes(int len)
        {
            var res = new byte[8];
            for(int i = 0; i < 4; i++)
            {
                res[ 7 - i ] = (byte)((len & (0xFF << (8*i))) >> (8*i));
            }
            return res;
        }
        var len = bytes.Length;
        var lenBytes = getLenBytes(len) ;
        Debug.Log($"message length is {len} and its bytes are: {lenBytes.Select(b => $"{b:X}").ToCommaSeparatedString()}");
        this.socket.Send(lenBytes);
        this.socket.Send(bytes);

    }
    byte[] Recv()
    {
        UInt64 getBytesLen(byte[] bytes)
        {
            UInt64 res = 0;
            for(int i = 0; i < 8; i++)
            {
                var currentByte = bytes[7-i];
                res = res | (((ulong)currentByte) << (8 * i));
            }
            return res;
        }
        byte[] lenBuf = new byte[8];
        this.socket.Receive(lenBuf);
        var len = getBytesLen(lenBuf);
        var msgBuf = new byte[len];
        this.socket.Receive(msgBuf);
        return msgBuf;
    }

    public GameAction SendStateAndRecvAction(GameState state, float reward)
    {
        void sendState()
        {
            var toSend = new GameStateAndReward(){
                Reward = reward,
                State = state
            };
            var toSendBytes = UTF8Encoding.UTF8.GetBytes( JsonConvert.SerializeObject(toSend));
            this.Send(toSendBytes);
        }
        GameAction recvAction()
        {
            var msgBuf = this.Recv();
            var msgStr = UTF8Encoding.UTF8.GetString(msgBuf);
            Debug.Log($"input message is: {msgStr}");
            var parsedValue = (GameAction)JsonConvert.DeserializeObject(msgStr, typeof(GameAction));
            Debug.Log($"parsed input is: {parsedValue}");
            return parsedValue;
        }
        try
        {
            sendState();
            return recvAction();
        }
        catch(Exception e)
        {
#if UNITY_EDITOR
        UnityEditor.EditorApplication.ExitPlaymode();
#else
            Application.Quit();
#endif
            throw;
        }
    }
}

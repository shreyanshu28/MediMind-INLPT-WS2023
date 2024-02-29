import React from 'react';
import brain from './assets/brain.png';
import plus from './assets/add-30.png';
import './App.css';
import msgIcon from './assets/message.svg';
import sendBtn from './assets/send.svg';
import userIcon from './assets/user-icon.png';

function App() {
    // TODO: Add state for input
    // const [input, setInput] = React.useState('');
    // const handleSend = () => {
    //     console.log(input);
  return (
    <div className="App">
        <div className="sideBar">
            <div className="upperSide">
                <div className="upperSideTop"><img src={brain} className="logo" alt=""/><span className="brand"><strong>Med-Mind</strong></span></div>
                <button className="midBtn"><img src={plus} alt="New Chat" className="addBtn"/>New Chat</button>
                <div className="upperSideBottom">
                    <button className="query"><img src={msgIcon} alt=""/>What is pubMed?</button>
                    <button className="query"><img src={msgIcon} alt=""/>define what is intelligence</button>
                </div>
            </div>
            <div className="lowerSide">
                <div className="listItems">Natural Language Processing Project</div>
            </div>
        </div>
        <div className="main">
            <div className="chats">
                <div className="chat">
                    <img className="chatImg" src={userIcon} alt=""/><p className="txt">Lorem ipsum dolor sit amet, consectetur adipisicing elit. A accusantium adipisci amet, assumenda culpa cupiditate delectus ducimus eius exercitationem inventore ipsa ipsam ipsum iste itaque minima molestiae molestias nam necessitatibus nisi officia porro provident quaerat quam qui quis quisquam quos ratione reprehenderit saepe sunt tenetur unde veniam veritatis! Accusantium alias at commodi consequatur deserunt eligendi error fuga illum iusto minima minus modi natus nesciunt nisi omnis provident sapiente sequi, similique veniam voluptate? Cum distinctio enim fugiat perspiciatis reprehenderit! A deleniti exercitationem facilis laboriosam minima modi possimus provident soluta, totam? Aliquam aperiam atque aut deserunt fugit harum id molestiae reprehenderit tempore.</p>
                </div>
                <div className="chat bot">
                    <img className="chatImg" src={brain} alt=""/><p className="txt">Lorem ipsum dolor sit amet, consectetur adipisicing elit. Ducimus, mollitia, quos. Ab aliquam animi, at blanditiis consectetur deleniti enim facere libero nulla numquam officia quos sequi vitae. Aut consectetur cumque dolorum hic illo incidunt, mollitia numquam odio quo quos reprehenderit sapiente sit soluta veniam voluptatem! Alias aliquid aut consequuntur culpa dignissimos dolor dolore doloremque, est eum eveniet exercitationem fuga hic id ipsam magnam maiores nulla numquam quisquam quo repellat reprehenderit suscipit temporibus vitae. A animi aspernatur beatae blanditiis commodi delectus, dolores dolorum enim eum laboriosam maiores nam odit pariatur quis tempora vel velit? Eius esse et illo quasi? Ad, voluptatem?</p>
                </div>
            </div>
            <div className="chatsfooter">
                <div className="inp">
                <input type="text" placeholder="Type a question..."/><button className="send"><img src={sendBtn} alt=""/></button>
                </div>
            </div>
        </div>

    </div>

  );
}

export default App;

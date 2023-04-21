import React, { useState, useEffect } from 'react';

function App() {
  const [image1, setImage1] = useState(null);
  const [image2, setImage2] = useState(null);
  const [combinedImage, setCombinedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [time, setTime] = useState(0);

  useEffect(() => {
    let intervalId;
    if (loading) {
      intervalId = setInterval(() => {
        setTime((time) => time + 1);
      }, 1000);
    } else {
      clearInterval(intervalId);
    }
    return () => clearInterval(intervalId);
  }, [loading]);

  const handleImage1Change = (event) => {
    setImage1(event.target.files[0]);
  };

  const handleImage2Change = (event) => {
    setImage2(event.target.files[0]);
  };

  const handleCombineClick = () => {
    setLoading(true);
    setStatus('Combining...');

    const formData = new FormData();
    formData.append('style', image1);
    formData.append('content', image2);

    fetch('https://styletransfer-kschmxl.pythonanywhere.com/transfer', {
      method: 'POST',
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        const imgData = data['image'];
        const imgSrc = 'data:image/jpeg;base64,' + imgData;
        setCombinedImage(imgSrc);
        setStatus('Combined successfully!');
      })
      .catch((error) => {
        console.error('Error:', error);
        setStatus('Error combining images.');
      })
      .finally(() => setLoading(false));
  };

return (
  <div style={{ maxWidth: '600px', margin: '0 auto' }}>
    <h1 style={{ textAlign: 'center', marginBottom: '2rem' }}>
      Image Style Transfer
    </h1>
    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
      <div style={{ width: '45%' }}>
        <label htmlFor="image1" style={{ display: 'block', marginBottom: '0.5rem' }}>
          Style Image:
        </label>
        <input type="file" id="image1" onChange={handleImage1Change} />
        {image1 && (
          <img
            src={URL.createObjectURL(image1)}
            alt="Style"
            style={{ marginTop: '1rem', width: '100%', objectFit: 'contain' }}
          />
        )}
      </div>
      <div style={{ width: '45%' }}>
        <label htmlFor="image2" style={{ display: 'block', marginBottom: '0.5rem' }}>
          Content Image:
        </label>
        <input type="file" id="image2" onChange={handleImage2Change} />
        {image2 && (
          <img
            src={URL.createObjectURL(image2)}
            alt="Content"
            style={{ marginTop: '1rem', width: '100%', objectFit: 'contain' }}
          />
        )}
      </div>
    </div>
    <button
      onClick={handleCombineClick}
      disabled={!image1 || !image2 || loading}
      style={{
        display: 'block',
        margin: '2rem auto',
        backgroundColor: '#4caf50',
        color: '#fff',
        border: 'none',
        padding: '0.5rem 1rem',
        borderRadius: '0.25rem',
        fontSize: '1rem',
        cursor: 'pointer',
      }}
    >
      {loading ? 'Combining...' : 'Combine Images'}
    </button>
    {status && (
      <p style={{ textAlign: 'center', fontWeight: 'bold' }}>{status}</p>
    )}
    {combinedImage && (
      <div style={{ marginTop: '2rem' }}>
        <h2 style={{ textAlign: 'center' }}>Combined Image:</h2>
        <img
          src={combinedImage}
          alt="Combined"
          style={{ display: 'block', margin: '0 auto', maxWidth: '100%' }}
        />
      </div>
    )}
    {loading && (
      <p style={{ textAlign: 'center', fontWeight: 'bold' }}>
        Time elapsed: {time} seconds
      </p>
    )}
  </div>
);
}

export default App;
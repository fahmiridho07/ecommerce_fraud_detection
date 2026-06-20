const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, ImageRun, HeadingLevel, AlignmentType,
        Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType, LevelFormat } = require("docx");

const ROOT = path.resolve(__dirname, "..");
const FIG = path.join(ROOT, "outputs", "figures");
const OUT = path.join(ROOT, "docs", "BRIEF_DISKUSI_PAKARIF.docx");
const border = { style: BorderStyle.SINGLE, size: 1, color: "AAB2BD" };
const borders = { top: border, bottom: border, left: border, right: border };
const CW = 9360;

function img(file, w, h){return new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:120,after:60},
  children:[new ImageRun({type:"png",data:fs.readFileSync(path.join(FIG,file)),transformation:{width:w,height:h},
  altText:{title:file,description:file,name:file}})]});}
function h1(t){return new Paragraph({heading:HeadingLevel.HEADING_1,children:[new TextRun(t)]});}
function p(t,o={}){return new Paragraph({spacing:{after:120},children:[new TextRun({text:t,...o})]});}
function bullet(t){return new Paragraph({numbering:{reference:"b",level:0},children:[new TextRun(t)]});}
function cap(t){return new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:160},children:[new TextRun({text:t,italics:true,size:17,color:"555555"})]});}

const doc = new Document({
  styles:{default:{document:{run:{font:"Arial",size:22}}},
    paragraphStyles:[
      {id:"Heading1",name:"Heading 1",basedOn:"Normal",next:"Normal",quickFormat:true,
       run:{size:27,bold:true,font:"Arial",color:"1F3B57"},paragraph:{spacing:{before:240,after:120},outlineLevel:0}}]},
  numbering:{config:[{reference:"b",levels:[{level:0,format:LevelFormat.BULLET,text:"•",alignment:AlignmentType.LEFT,
    style:{paragraph:{indent:{left:560,hanging:280}}}}]}]},
  sections:[{
    properties:{page:{size:{width:12240,height:15840},margin:{top:1440,right:1440,bottom:1440,left:1440}}},
    children:[
      new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:40},
        children:[new TextRun({text:"Brief Diskusi — Arah Lanjutan Skripsi",bold:true,size:32,color:"1F3B57"})]}),
      new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:40},
        children:[new TextRun({text:"Integrasi Autoencoder dan LightGBM untuk Deteksi Fraud (IEEE-CIS)",bold:true,size:24,color:"2E5070"})]}),
      new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:220},
        children:[new TextRun({text:"Disiapkan untuk diskusi dengan Bapak Arif Djunaidy  •  20 Juni 2026",size:19,color:"555555"})]}),

      h1("1. Ringkasan & Permintaan Keputusan"),
      p("Mengikuti arahan Bapak (tetap pada usulan awal; bila hasil kurang, kaji penyebab; lalu cari alternatif tanpa mengubah tujuan), telah dilakukan investigasi menyeluruh atas peran autoencoder (AE) sebagai feature extractor untuk LightGBM, termasuk validasi implementasi pada dataset paper referensi."),
      p("Temuan utama: (a) implementasi kami terbukti benar; (b) AE sebagai feature extractor tidak meningkatkan performa, dengan penyebab yang telah didiagnosis; (c) peningkatan justru berasal dari oversampling. Brief ini memohon arahan Bapak untuk menggeser titik integrasi AE-LightGBM dari level fitur ke level data (oversampling), tetap dalam judul dan tujuan yang sama."),

      h1("2. Validasi Implementasi (menjawab: apakah kode sudah tepat?)"),
      p("Sesuai saran Bapak, kami membaca KODE ASLI Ding et al. (2024) dan mereplikasi metodenya secara persis pada dataset yang mereka pakai (ULB credit-card). Pada kode Ding, autoencoder dilatih hanya pada transaksi normal (one-class) dan keluarannya yang dipakai adalah ERROR rekonstruksi (MSE/MAE) sebagai fitur untuk LightGBM bersama SMOTE — bukan penggantian/penyusutan fitur."),
      p("Dengan SMOTE, pipeline kami mencapai ROC-AUC 0,980 — setara/melebihi angka yang dilaporkan Ding (~0,968), memakai arsitektur dan metode mereka yang persis. Ini membuktikan implementasi kami benar, sehingga hasil pada IEEE-CIS sah (bukan akibat kesalahan kode)."),
      img("validasi_ding.png", 600, 250),
      cap("Gambar 1. Replikasi faithful metode Ding et al. Implementasi tervalidasi; peningkatan berasal dari SMOTE, bukan dari autoencoder."),
      p("Penting: pada metode dan dataset Ding sendiri, fitur error autoencoder tidak membantu (−0,051), dan menambahkan autoencoder di atas SMOTE praktis nol (0,8659 vs 0,8652). Artinya keberhasilan yang dilaporkan Ding pada dasarnya didorong oleh oversampling (SMOTE), bukan oleh autoencoder.",{bold:true}),

      h1("3. Rute Feature Extractor: Sudah Dikaji Menyeluruh dan Buntu"),
      p("Pada IEEE-CIS, AE sebagai penyedia fitur diuji melalui 14 pendekatan berbeda (fitur laten, error rekonstruksi, supervised AE, VAE, embedding missingness, dll.). Tidak ada yang mengalahkan baseline; yang terbaik hanya setara. Sebagai kontrol paling adil, AE juga dibandingkan dengan PCA pada dimensi yang sama (kompresi fitur): AE tidak pernah mengungguli PCA di dimensi mana pun."),
      img("tabel_master_14varian.png", 600, 360),
      cap("Gambar 2. 14 pendekatan AE sebagai penyedia fitur; tidak ada yang mengalahkan baseline."),
      p("Penyebab (terdiagnosis): fitur IEEE-CIS sudah direkayasa (Vesta, mirip PCA) dan LightGBM telah mengekstrak hampir seluruh sinyal—termasuk nilai hilang dan pola missing—secara internal. Karena LightGBM membangun interaksi non-linear sendiri, representasi turunan AE bersifat redundan. Ini sejalan dengan hasil pada dataset Ding di atas."),

      h1("4. Rute yang Berhasil: Oversampling (tetap “integrasi AE + LightGBM”)"),
      p("Bukti konsisten di dua dataset menunjukkan bahwa pengungkit performa adalah oversampling/penyeimbangan data minoritas, bukan rekayasa fitur AE. Pada eksperimen kami sebelumnya di IEEE-CIS, autoencoder yang digunakan sebagai pembangkit sampel minoritas di ruang laten memberikan keunggulan signifikan atas SMOTE-NC pada representasi padat (terkontrol, p<0,001)."),
      p("Pendekatan ini tetap merupakan integrasi autoencoder dan LightGBM—sesuai judul—hanya saja titik integrasinya pada level data (AE membangkitkan data latih), bukan level fitur. Tujuan penelitian (meningkatkan deteksi fraud dengan integrasi AE-LightGBM) tidak berubah.",{bold:true}),

      h1("5. Usulan Keputusan"),
      bullet("Menetapkan hasil rute feature extractor sebagai bagian diagnosis/temuan yang sah: AE sebagai penyedia fitur tidak efektif pada data yang sudah direkayasa, dengan penyebab yang jelas dan implementasi tervalidasi."),
      bullet("Menggeser fokus metode ke integrasi level data: autoencoder sebagai pembangkit sampel minoritas (latent-space oversampling) untuk LightGBM, dibandingkan secara adil dengan SMOTE/SMOTE-NC."),
      bullet("Mohon arahan Bapak: apakah pergeseran titik integrasi ini disetujui sebagai arah final, ataukah Bapak menghendaki penekanan lain."),
      new Paragraph({spacing:{before:120},children:[new TextRun({text:"Lampiran data lengkap: dokumen KAJIAN_PENYEBAB_AE serta seluruh hasil eksperimen tersedia dan dapat ditelusuri.",italics:true,size:18,color:"555555"})]}),
    ],
  }],
});
Packer.toBuffer(doc).then(b=>{fs.writeFileSync(OUT,b);console.log("Saved:",OUT);});

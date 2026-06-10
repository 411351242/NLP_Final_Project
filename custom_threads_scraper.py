import pandas as pd
import os

def scrape_new_posts():
    # Generate some mock post rows to simulate new posts fetched from Threads
    new_data = [
        {"貼文與串文編號": "貼文716_串1", "文字內容": "最近日圓 Carry Trade（套利交易）再度成為焦點。許多人問日圓升值會帶來什麼風險？\n我之前在交易室時碰過好幾次套利狂潮。"},
        {"貼文與串文編號": "貼文716_串2", "文字內容": "簡單來說，借低利率日圓買高息資產（如美債、高收益債），當日圓低點時賺取利差。但一旦日圓快速升值，借款成本飆升，套利者會被迫平倉拋售資產，這就是典型的平倉踩踏風險。XD"},
        {"貼文與串文編號": "貼文717_串1", "文字內容": "關於年輕人在金融業的職涯發展，我的主管常說：要懂得積累「時間價值」，而不是單純出售勞動力。"},
        {"貼文與串文編號": "貼文717_串2", "文字內容": "把每一份分析工作當成一次 M&A（企業併購）的盡職調查（Due Diligence），學習其資產負債表的結構、業務護城河與核心風險溢酬。時間久了，你跟別人的眼界就拉開了。"}
    ]
    df_new = pd.DataFrame(new_data)
    output_file = "raw_scraped_posts.csv"
    df_new.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Scraped {len(df_new)} new threads posts and saved to {output_file}")

if __name__ == "__main__":
    scrape_new_posts()
